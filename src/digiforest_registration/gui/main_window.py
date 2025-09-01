from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QThread

from digiforest_registration.gui.vtk_pointcloud_viewer import VTKPointCloud
from digiforest_registration.gui.tile_viewer import ShapeCanvas
from digiforest_registration.gui.pipeline_worker import PipelineWorker


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
        self._thread = QThread(self)
        self._worker = PipelineWorker(self.registration_args)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._thread.start()

        # Connect signals and slots
        self._worker.registration_finished.connect(self._handle_registration_finished)

    def _handle_registration_finished(self):
        self.progressBar.setValue(
            100 * self._worker.num_cloud_processed / self._worker.num_clouds
        )

    def _shutdown_worker(self):
        """Cooperatively stop the worker thread if it's running."""
        if self._thread and self._thread.isRunning():
            self._worker.stop()
            self._thread.requestInterruption()
            self._thread.quit()
            self._thread.wait()

    def closeEvent(self, e):
        self._shutdown_worker()
        super().closeEvent(e)

    def on_open(self):
        _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open file", "", "All Files (*.*)"
        )

    def on_about(self):
        QtWidgets.QMessageBox.information(
            self,
            "About",
            "Demo app: PyQt main window with menu/toolbar, a 2D drawing canvas,\n"
            "a bottom-left tab widget, and a VTK 3D point cloud viewer.",
        )
