import random
from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRectF, QPointF

try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    import vtkmodules.all as vtk
except Exception:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    import vtk as vtk


# -------------------------
# Programmatic 2D Shape Canvas
# -------------------------
class ShapeCanvas(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.shapes = []  # list of (type, params, color, pen_width)
        self.setMinimumHeight(200)

    def add_rectangle(self, x, y, w, h, color=Qt.black, pen_width=2):
        self.shapes.append(("rect", (x, y, w, h), QColor(color), pen_width))
        self.update()

    def add_ellipse(self, x, y, w, h, color=Qt.black, pen_width=2):
        self.shapes.append(("ellipse", (x, y, w, h), QColor(color), pen_width))
        self.update()

    def add_line(self, x1, y1, x2, y2, color=Qt.black, pen_width=2):
        self.shapes.append(("line", (x1, y1, x2, y2), QColor(color), pen_width))
        self.update()

    def clear_shapes(self):
        self.shapes.clear()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        for shape_type, params, color, pen_width in self.shapes:
            pen = QPen(color)
            pen.setWidth(pen_width)
            painter.setPen(pen)

            if shape_type == "rect":
                x, y, w, h = params
                painter.drawRect(QRectF(x, y, w, h))
            elif shape_type == "ellipse":
                x, y, w, h = params
                painter.drawEllipse(QRectF(x, y, w, h))
            elif shape_type == "line":
                x1, y1, x2, y2 = params
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))


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
