import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal
from aerial_terrestrial_registration.utils import (
    check_registration_inputs_validity,
    get_tiles_conf_file,
    CloudIO,
)
from aerial_terrestrial_registration.utils import crop_cloud, crop_cloud_to_size

from aerial_terrestrial_registration.registration.registration import (
    Registration,
    RegistrationResult,
)
from aerial_terrestrial_registration.registration.registration_io import (
    save_registered_clouds,
    save_posegraph,
)
from aerial_terrestrial_registration.utils import ExperimentLogger
from multiprocessing import Queue, Process
from logging.handlers import QueueHandler, QueueListener
import threading
import logging


def run_registration_process(
    args,
    uav_cloud_filename: str,
    mls_cloud_filename: str,
    mls_cloud_folder: str,
    cloud_io: CloudIO,
    log_queue: Queue,
    output_queue: Queue,
    logger: ExperimentLogger,
):

    # Set up logging queue
    qh = QueueHandler(log_queue)
    root = logger.logger
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(qh)

    # Loading the data
    uav_cloud = cloud_io.load_cloud(str(uav_cloud_filename))
    original_mls_cloud = cloud_io.load_cloud(str(mls_cloud_filename))

    # cropping input clouds
    if args.crop_mls_cloud:
        mls_cloud = crop_cloud_to_size(original_mls_cloud, size=30)
    else:
        mls_cloud = original_mls_cloud
    cropped_uav_cloud = crop_cloud(uav_cloud, mls_cloud, padding=20)

    cropped_uav_cloud = cropped_uav_cloud
    mls_cloud_folder = mls_cloud_folder
    registration = Registration(
        cropped_uav_cloud,
        mls_cloud,
        cloud_io,
        args.ground_segmentation_method,
        args.correspondence_matching_method,
        args.mls_feature_extraction_method,
        args.icp_fitness_score_threshold,
        args.min_distance_between_peaks,
        args.max_number_of_clique,
        logger,
        correspondence_graph_distance_threshold=args.correspondence_graph_distance_threshold,
        maximum_rotation_offset=args.maximum_rotation_offset,
        debug=args.debug,
    )
    success = registration.registration()
    save_registered_clouds(
        cloud_io,
        registration,
        mls_cloud_filename,
        original_mls_cloud,
        args.mls_registered_cloud_folder,
    )
    output_queue.put(success)
    output_queue.put(registration.transform)
    output_queue.put(registration.best_icp_fitness_score)


class RegistrationPipelineWorker(QObject):
    new_cloud = pyqtSignal()
    registration_finished = pyqtSignal()

    def __init__(self, args, logger: ExperimentLogger, cloud_io: CloudIO):
        super().__init__()
        self.args = args
        self.stop_event = threading.Event()
        self.registration_process = None
        self.output_queue = Queue()
        self.logger = logger
        self.cloud_io = cloud_io
        self.last_cloud = None
        self.num_clouds = 0
        self.num_cloud_processed = 0
        self.handler = logging.StreamHandler()
        self.handler.setLevel(logging.DEBUG)

    def stop(self):
        # Signal the thread to stop
        # Terminate the process if it's running
        self.stop_event.set()
        if self.registration_process is not None:
            self.registration_process.terminate()
            self.registration_process.join()

    def run(self):
        args = self.args
        self.num_clouds = 0
        self.num_cloud_processed = 0
        log_queue = Queue()
        listener = QueueListener(log_queue, self.handler)
        listener.start()
        try:
            # Check validity of inputs
            (
                mls_cloud_filenames,
                mls_cloud_folder,
                uav_cloud_filename,
            ) = check_registration_inputs_validity(args)

            self.num_clouds = len(mls_cloud_filenames)
            registration_results = {}
            failures = []
            successes = []
            for mls_cloud_filename in mls_cloud_filenames:
                self.logger.set_leaf_logging_folder(mls_cloud_filename.stem)
                self.registration_process = Process(
                    target=run_registration_process,
                    args=(
                        args,
                        uav_cloud_filename,
                        mls_cloud_filename,
                        mls_cloud_folder,
                        self.cloud_io,
                        log_queue,
                        self.output_queue,
                        self.logger,
                    ),
                )

                self.registration_process.start()
                while (
                    not self.stop_event.is_set()
                    and self.registration_process.is_alive()
                ):
                    try:
                        self.new_data.emit()
                    except Exception:
                        continue

                if self.stop_event.is_set():
                    # Stopping the thread
                    listener.stop()
                    break
                self.registration_process.join()
                self.num_cloud_processed += 1
                self.registration_finished.emit()

                # save results
                success = self.output_queue.get()
                transform = self.output_queue.get()
                best_icp_fitness_score = self.output_queue.get()

                result = RegistrationResult()
                result.transform = transform
                result.success = success
                result.icp_fitness = best_icp_fitness_score
                registration_results[mls_cloud_filename.name] = result

                if not success:
                    failures.append((mls_cloud_filename.name, (best_icp_fitness_score)))
                else:
                    successes.append(
                        (mls_cloud_filename.name, (best_icp_fitness_score))
                    )

            self.logger.info(f"Total number of failures: {len(failures)}")
            self.logger.info(f"Total number of clouds: {len(mls_cloud_filenames)}")
            self.logger.info(f"Failures: {failures}")
            self.logger.info(f"Successes: {successes}")

            # end of the processing
            if not args.tiles and not args.save_pose_graph:
                return

            tiles_conf_file = get_tiles_conf_file(self.args)
            noise_matrix = np.array(args.noise_matrix, dtype=np.float32)
            save_posegraph(
                noise_matrix,
                tiles_conf_file,
                args.save_pose_graph,
                args.mls_registered_cloud_folder,
                args.pose_graph_file,
                registration_results,
                mls_cloud_folder,
                self.cloud_io.offset,
                args.icp_fitness_score_threshold,
            )
        except Exception as e:
            print(f"Registration failed: {e}")
