from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QThread

from digiforest_registration.gui.vtk_pointcloud_viewer import VTKPointCloud
from digiforest_registration.gui.image_viewer import ImageWidget
from digiforest_registration.gui.pipeline_worker import PipelineWorker
from digiforest_registration.gui.log_tree_widget import FileTreeWidget
from digiforest_registration.utils import ExperimentLogger

import os
import numpy as np

from digiforest_registration.utils.cloud_io import CloudIO


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, registration_args):
        super().__init__()
        self.args = registration_args
        self.args.debug = False  # disable debug mode in GUI

        logging_dir = self.args.logging_dir
        if self.args.logging_dir is None:
            logging_dir = "./logs"
        logging_dir = os.path.join(logging_dir)
        self.logger = ExperimentLogger(base_dir=logging_dir, log_pointclouds=True)

        if self.args.offset is not None and len(self.args.offset) == 3:
            offset = np.array(
                [
                    self.args.offset[0],
                    self.args.offset[1],
                    self.args.offset[2],
                ],
                dtype=np.float32,
            )
        else:
            # default offset
            offset = np.array([0, 0, 0], dtype=np.float32)

        self.cloud_io = CloudIO(
            offset, logger=None, downsample_cloud=self.args.downsample_cloud
        )

        # Load the UI file
        # TODO improve path
        uic.loadUi("../src/digiforest_registration/gui/main_window.ui", self)

        # Replace placeholders with custom widgets
        self.image_viewer = ImageWidget()
        self.canvasPlaceholder.layout().addWidget(self.image_viewer)

        self.vtk_viewer = VTKPointCloud()
        self.vtkLayout.addWidget(self.vtk_viewer)

        self.logTreeWidget = FileTreeWidget(
            root_path=self.logger.current_logging_directory()
        )
        self.tabLogs.layout().addWidget(self.logTreeWidget)
        self.logTreeWidget.fileChecked.connect(
            lambda filename: self.vtk_viewer.load_pointcloud(filename, self.cloud_io)
            if filename.endswith(".ply")
            else self.image_viewer.load_image(filename)
        )

        self.logTreeWidget.fileUnChecked.connect(
            lambda filename: self.vtk_viewer.delete_pointcloud(filename)
            if filename.endswith(".ply")
            else self.image_viewer.delete_image(filename)
        )

        # self.outputTreeWidget = FileTreeWidget(
        #     root_path=self.args.mls_registered_cloud_folder
        # )
        # self.tabOutputs.layout().addWidget(self.outputTreeWidget)
        # self.outputTreeWidget.fileChecked.connect(
        #     lambda filename: self.vtk_viewer.load_pointcloud(filename, self.cloud_io)
        # )

        self.inputTreeWidget = FileTreeWidget(root_path=self.args.mls_cloud_folder)
        self.tabInputs.layout().addWidget(self.inputTreeWidget)
        self.inputTreeWidget.fileChecked.connect(
            lambda filename: self.vtk_viewer.load_pointcloud(filename, self.cloud_io)
            if filename.endswith(".ply")
            else self.image_viewer.load_image(filename)
        )

        self.inputTreeWidget.fileUnChecked.connect(
            lambda filename: self.vtk_viewer.delete_pointcloud(filename)
            if filename.endswith(".ply")
            else self.image_viewer.delete_image(filename)
        )

        # Connect menu/toolbar actions
        self.actionRunRegistration.triggered.connect(self.start_registration)
        self.actionReset3D.triggered.connect(self.vtk_viewer.reset_camera)
        self.actionOpen.triggered.connect(self.on_open)
        self.actionAbout.triggered.connect(self.on_about)
        self.actionQuit.triggered.connect(self.close)

        self._thread = None

        self.statusBar().showMessage("Ready")
        self.progressBar = QtWidgets.QProgressBar()
        self.statusBar().addPermanentWidget(self.progressBar)
        self.progressBar.setGeometry(30, 40, 200, 25)
        self.progressBar.setValue(0)

    def start_registration(self):
        self.progressBar.setValue(0)
        self._thread = QThread(self)
        self._worker = PipelineWorker(self.args, self.logger, self.cloud_io)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._thread.start()

        # Connect signals and slots
        self._worker.registration_finished.connect(self._handle_registration_finished)

    def _handle_registration_finished(self):
        self.progressBar.setValue(
            100 * self._worker.num_cloud_processed / self._worker.num_clouds
        )
        self.logTreeWidget.update_view()
        self.outputTreeWidget.update_view()

    def _shutdown_worker(self):
        """Cooperatively stop the worker thread if it's running."""
        if self._thread and self._thread.isRunning():
            self._worker.stop()
            self._thread.requestInterruption()
            self._thread.quit()
            self._thread.wait()

    def closeEvent(self, e):
        self._shutdown_worker()
        self.logger.delete_all_logs()
        super().closeEvent(e)

    def on_open(self):
        _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open file", "", "All Files (*.*)"
        )

    def on_about(self):
        QtWidgets.QMessageBox.information(
            self, "About", "Point Cloud Registration Tool\n"
        )
