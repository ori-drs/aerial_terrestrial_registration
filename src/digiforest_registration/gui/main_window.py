import random
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPen

try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    import vtkmodules.all as vtk
except Exception:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    import vtk as vtk


# -------------------------
# Simple 2D drawing canvas
# -------------------------
class DrawingCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.paths = []  # list[list[QPoint]]
        self.current_path = []  # list[QPoint]
        self.pen_width = 2

    def clear(self):
        self.paths.clear()
        self.current_path.clear()
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.current_path = [event.pos()]
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.current_path:
            self.current_path.append(event.pos())
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.current_path:
            self.paths.append(self.current_path[:])
            self.current_path = []
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        pen = QPen()
        pen.setWidth(self.pen_width)
        painter.setPen(pen)

        # Draw finished paths
        for path in self.paths:
            for i in range(1, len(path)):
                painter.drawLine(path[i - 1], path[i])

        # Draw current path
        for i in range(1, len(self.current_path)):
            painter.drawLine(self.current_path[i - 1], self.current_path[i])


# -------------------------
# VTK 3D point cloud widget
# -------------------------
class VTKPointCloud(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.interactor.Initialize()

        self._actor = None
        self._setup_scene()

    def _setup_scene(self):
        self.renderer.SetBackground(0.1, 0.1, 0.12)
        self.add_demo_points()
        self.renderer.ResetCamera()

    def add_demo_points(self, n=2000):
        # Generate some random XYZ points as a demo
        points = vtk.vtkPoints()
        for _ in range(n):
            x = random.uniform(-10, 10)
            y = random.uniform(-10, 10)
            z = random.uniform(-2, 2)
            points.InsertNextPoint(x, y, z)

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)

        glyph = vtk.vtkVertexGlyphFilter()
        glyph.SetInputData(poly)
        glyph.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(3)

        if self._actor:
            self.renderer.RemoveActor(self._actor)
        self._actor = actor
        self.renderer.AddActor(actor)
        self.vtk_widget.GetRenderWindow().Render()

    def reset_camera(self):
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()


# -------------------------
# Main window
# -------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Load the UI file
        # TODO improve path
        uic.loadUi("../src/digiforest_registration/gui/main_window.ui", self)

        # Replace placeholders with custom widgets
        self.canvas = DrawingCanvas()
        self.canvasPlaceholder.layout().addWidget(self.canvas)

        self.vtk_viewer = VTKPointCloud()
        self.vtkLayout.addWidget(self.vtk_viewer)

        # Connect menu/toolbar actions
        self.actionClear2D.triggered.connect(self.canvas.clear)
        self.actionReset3D.triggered.connect(self.vtk_viewer.reset_camera)
        self.actionOpen.triggered.connect(self.on_open)
        self.actionSave.triggered.connect(self.on_save)
        self.actionAbout.triggered.connect(self.on_about)
        self.actionQuit.triggered.connect(self.close)

        self.statusBar().showMessage("Ready")

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
