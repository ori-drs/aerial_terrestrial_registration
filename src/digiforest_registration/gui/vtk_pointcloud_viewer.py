import random
from PyQt5 import QtWidgets

try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    import vtkmodules.all as vtk
except Exception:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    import vtk as vtk


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
