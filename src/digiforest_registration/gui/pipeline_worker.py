import numpy as np
import os

from PyQt5.QtCore import QObject, pyqtSignal
from digiforest_registration.utils import (
    check_registration_inputs_validity,
    CloudIO,
)
from digiforest_registration.utils import crop_cloud, crop_cloud_to_size

# from digiforest_registration.registration.registration import Registration
from multiprocessing import Queue, Process
import threading


def run_registration_process(
    args,
    uav_cloud_filename: str,
    mls_cloud_filename: str,
    mls_cloud_folder: str,
    queue: Queue,
):

    # Loading the data
    if args.offset is not None and len(args.offset) == 3:
        offset = np.array(
            [
                args.offset[0],
                args.offset[1],
                args.offset[2],
            ],
            dtype=np.float32,
        )
    else:
        # default offset
        offset = np.array([0, 0, 0], dtype=np.float32)

    cloud_io = CloudIO(offset, logger=None, downsample_cloud=args.downsample_cloud)
    uav_cloud = cloud_io.load_cloud(str(uav_cloud_filename))
    original_mls_cloud = cloud_io.load_cloud(str(mls_cloud_filename))

    # cropping input clouds
    if args.crop_mls_cloud:
        mls_cloud = crop_cloud_to_size(original_mls_cloud, size=30)
    else:
        mls_cloud = original_mls_cloud
    cropped_uav_cloud = crop_cloud(uav_cloud, mls_cloud, padding=20)

    logging_dir = args.logging_dir
    if args.logging_dir is None:
        logging_dir = "./logs"
    logging_dir = os.path.join(logging_dir, mls_cloud_filename.stem)

    # if (
    #     args.mls_registered_cloud_folder is not None
    #     and args.tiles_conf_file is not None
    # ):
    #     tile_config_reader = TileConfigReader(
    #         args.tiles_conf_file, offset
    #     )

    cropped_uav_cloud = cropped_uav_cloud
    mls_cloud_folder = mls_cloud_folder
    queue = queue
    # registration = Registration(
    #     cropped_uav_cloud,
    #     mls_cloud,
    #     args.ground_segmentation_method,
    #     args.correspondence_matching_method,
    #     args.mls_feature_extraction_method,
    #     args.icp_fitness_score_threshold,
    #     args.min_distance_between_peaks,
    #     args.max_number_of_clique,
    #     logging_dir,
    #     correspondence_graph_distance_threshold=args.correspondence_graph_distance_threshold,
    #     maximum_rotation_offset=args.maximum_rotation_offset,
    #     debug=args.debug,
    # )

    # registration = registration
    print("End of process")


class PipelineWorker(QObject):
    new_data = pyqtSignal()

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.stop_event = threading.Event()
        self.registration_process = None
        self.queue = Queue()
        self.last_cloud = None

    def stop(self):
        # Signal the thread to stop
        # Terminate the process if it's running
        self.stop_event.set()
        if self.registration_process is not None:
            self.registration_process.terminate()
            self.registration_process.join()

    def run(self):
        args = self.args
        try:
            print("Initialization...")
            # Check validity of inputs
            (
                mls_cloud_filenames,
                mls_cloud_folder,
                uav_cloud_filename,
            ) = check_registration_inputs_validity(args)

            for mls_cloud_filename in mls_cloud_filenames:
                self.registration_process = Process(
                    target=run_registration_process,
                    args=(
                        args,
                        uav_cloud_filename,
                        mls_cloud_filename,
                        mls_cloud_folder,
                        self.queue,
                    ),
                )

                self.registration_process.start()
                while (
                    not self.stop_event.is_set()
                    and self.registration_process.is_alive()
                ):
                    try:
                        _ = self.queue.get(timeout=0.5)
                        self.new_data.emit()
                    except Exception:  # queue.Empty
                        continue

                if self.stop_event.is_set():
                    break

        except Exception as e:
            print(f"Registration failed: {e}")
