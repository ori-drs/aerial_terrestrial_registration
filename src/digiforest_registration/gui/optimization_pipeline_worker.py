from PyQt5.QtCore import QObject, pyqtSignal
from digiforest_registration.utils import (
    check_optimization_inputs_validity,
    CloudIO,
)
from digiforest_registration.optimization.graph_optimization import (
    PoseGraphOptimization,
)
from digiforest_registration.optimization.io import (
    load_pose_graph,
    write_pose_graph,
    save_optimized_pointclouds,
)
from pathlib import Path
from digiforest_registration.utils import ExperimentLogger
from multiprocessing import Queue, Process
from logging.handlers import QueueHandler, QueueListener
import threading
import logging


def run_optimization_process(
    args,
    cloud_io: CloudIO,
    log_queue: Queue,
    logger: ExperimentLogger,
):

    # Set up logging queue
    qh = QueueHandler(log_queue)
    root = logger.logger
    root.setLevel(logging.DEBUG)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(qh)

    # run optimization
    mls_cloud_folder = check_optimization_inputs_validity(args)
    pose_graph_file = str(Path(args.mls_registered_cloud_folder) / "pose_graph.g2o")

    pose_graph = load_pose_graph(
        pose_graph_file,
        mls_cloud_folder,
        cloud_io,
        args.load_clouds,
        args.tiles,
    )

    optimizer = PoseGraphOptimization(pose_graph, False, args.load_clouds, logger)
    optimizer.optimize()

    # save results
    if args.optimized_cloud_output_folder is not None:
        pose_graph_output_file = (
            Path(args.optimized_cloud_output_folder) / "optimized_pose_graph.g2o"
        )
        write_pose_graph(pose_graph, str(pose_graph_output_file))

    # save the optmized clouds
    save_optimized_pointclouds(
        args.optimized_cloud_output_folder, args.load_clouds, pose_graph, cloud_io
    )


class OptimizationPipelineWorker(QObject):
    optimization_finished = pyqtSignal()

    def __init__(self, args, logger: ExperimentLogger, cloud_io: CloudIO):
        super().__init__()
        self.args = args
        self.stop_event = threading.Event()
        self.optimization_process = None
        self.logger = logger
        self.cloud_io = cloud_io
        self.handler = logging.StreamHandler()
        self.handler.setLevel(logging.DEBUG)

    def stop(self):
        # Signal the thread to stop
        # Terminate the process if it's running
        self.stop_event.set()
        if self.optimization_process is not None:
            self.optimization_process.terminate()
            self.optimization_process.join()

    def run(self):
        args = self.args
        self.num_clouds = 0
        self.num_cloud_processed = 0
        log_queue = Queue()
        listener = QueueListener(log_queue, self.handler)
        listener.start()
        try:
            self.optimization_process = Process(
                target=run_optimization_process,
                args=(
                    args,
                    self.cloud_io,
                    log_queue,
                    self.logger,
                ),
            )

            self.optimization_process.start()
            while not self.stop_event.is_set() and self.optimization_process.is_alive():
                try:
                    self.new_data.emit()
                except Exception:
                    continue

                if self.stop_event.is_set():
                    # Stopping the thread
                    listener.stop()
                    break

            self.optimization_process.join()
            self.optimization_finished.emit()
            # end of the processing

        except Exception as e:
            print(f"Registration failed: {e}")
