import os
import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QThread

from digiforest_registration.gui.pointcloud_viewer import VTKPointCloud
from digiforest_registration.gui.tile_viewer import ShapeCanvas
from digiforest_registration.gui.pipeline_worker import PipelineWorker
from digiforest_registration.utils import (
    check_registration_inputs_validity,
    CloudIO,
)
from digiforest_registration.utils import crop_cloud, crop_cloud_to_size
from digiforest_registration.registration.registration import Registration


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, registration_args):
        super().__init__()
        self.registration_args = registration_args
        # Load the UI file
        # TODO improve path
        uic.loadUi("../src/digiforest_registration/gui/main_window.ui", self)

        # Replace placeholders with custom widgets
        self.canvas = ShapeCanvas()
        self.canvasPlaceholder.layout().addWidget(self.canvas)

        self.vtk_viewer = VTKPointCloud()
        self.vtkLayout.addWidget(self.vtk_viewer)

        # Connect menu/toolbar actions
        self.actionReset3D.triggered.connect(self.vtk_viewer.reset_camera)
        self.actionOpen.triggered.connect(self.on_open)
        self.actionSave.triggered.connect(self.on_save)
        self.actionAbout.triggered.connect(self.on_about)
        self.actionQuit.triggered.connect(self.close)

        self.statusBar().showMessage("Ready")

    def start_registration(self, registration_args):

        # Check validity of inputs
        (
            mls_cloud_filenames,
            mls_cloud_folder,
            uav_cloud_filename,
        ) = check_registration_inputs_validity(registration_args)

        # Loading the data
        if registration_args.offset is not None and len(registration_args.offset) == 3:
            offset = np.array(
                [
                    registration_args.offset[0],
                    registration_args.offset[1],
                    registration_args.offset[2],
                ],
                dtype=np.float32,
            )
        else:
            # default offset
            offset = np.array([0, 0, 0], dtype=np.float32)

        cloud_io = CloudIO(
            offset, logger=None, downsample_cloud=registration_args.downsample_cloud
        )
        uav_cloud = cloud_io.load_cloud(str(uav_cloud_filename))

        # if (
        #     registration_args.mls_registered_cloud_folder is not None
        #     and registration_args.tiles_conf_file is not None
        # ):
        #     tile_config_reader = TileConfigReader(
        #         registration_args.tiles_conf_file, offset
        #     )

        for mls_cloud_filename in mls_cloud_filenames:

            original_mls_cloud = cloud_io.load_cloud(str(mls_cloud_filename))

            # cropping input clouds
            if registration_args.crop_mls_cloud:
                mls_cloud = crop_cloud_to_size(original_mls_cloud, size=30)
            else:
                mls_cloud = original_mls_cloud
            cropped_uav_cloud = crop_cloud(uav_cloud, mls_cloud, padding=20)

            logging_dir = registration_args.logging_dir
            if registration_args.logging_dir is None:
                logging_dir = "./logs"
            logging_dir = os.path.join(logging_dir, mls_cloud_filename.stem)

            registration = Registration(
                cropped_uav_cloud,
                mls_cloud,
                registration_args.ground_segmentation_method,
                registration_args.correspondence_matching_method,
                registration_args.mls_feature_extraction_method,
                registration_args.icp_fitness_score_threshold,
                registration_args.min_distance_between_peaks,
                registration_args.max_number_of_clique,
                logging_dir,
                correspondence_graph_distance_threshold=registration_args.correspondence_graph_distance_threshold,
                maximum_rotation_offset=registration_args.maximum_rotation_offset,
                debug=registration_args.debug,
            )

            self._thread = QThread(self)
            self._worker = PipelineWorker(registration)
            self._worker.moveToThread(self._thread)

            self._thread.started.connect(self._worker.run)

    # -------- Actions --------
    def on_open(self):
        _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open file", "", "All Files (*.*)"
        )

    def on_save(self):
        _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save file", "", "PNG Image (*.png);;All Files (*.*)"
        )

    def on_about(self):
        QtWidgets.QMessageBox.information(
            self,
            "About",
            "Demo app: PyQt main window with menu/toolbar, a 2D drawing canvas,\n"
            "a bottom-left tab widget, and a VTK 3D point cloud viewer.",
        )
