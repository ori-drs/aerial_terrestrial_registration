import numpy as np
from vtk.util import numpy_support
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout, QApplication

from digiforest_registration.utils import CloudIO

try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    import vtkmodules.all as vtk
except Exception:
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    import vtk as vtk


class WaitingDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Loading...")
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(QLabel("Please wait, processing..."))


class VTKPointCloud(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        self.edl = None
        self.renderer = vtk.vtkRenderer()
        self.renderer.AutomaticLightCreationOn()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.interactor.Initialize()

        self._actors = {}
        self._setup_scene()

    def _setup_scene(self):
        self.renderer.SetBackground(1.0, 1.0, 1.0)
        self.renderer.ResetCamera()

    def delete_pointcloud(self, filename: str):
        if filename in self._actors:
            self.renderer.RemoveActor(self._actors[filename])
            self.vtk_widget.GetRenderWindow().Render()

    def load_pointcloud(self, filename: str, cloud_io: CloudIO):
        self.dialog = WaitingDialog()
        self.dialog.show()
        QApplication.processEvents()
        self.insert_pointcloud(filename, cloud_io)
        self.dialog.close()
        self.reset_camera()

    def insert_pointcloud(self, filename: str, cloud_io: CloudIO):
        if self.edl is not None:
            self.edl.ReleaseGraphicsResources(self.vtk_widget.GetRenderWindow())

        cloud = cloud_io.load_cloud(filename)
        xyz = cloud.point["positions"].numpy()

        points = vtk.vtkPoints()
        # points.SetData(numpy_support.numpy_to_vtk(xyz.astype(np.float32)))
        for i in range(xyz.shape[0]):
            points.InsertNextPoint(xyz[i, 0], xyz[i, 1], xyz[i, 2])

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)

        if "colors" in cloud.point:
            rgb = 255 * cloud.point["colors"].numpy()
            rgb = rgb.astype(np.uint8)
            vtk_colors = numpy_support.numpy_to_vtk(
                rgb, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
            )
            vtk_colors.SetName("Colors")
            polydata.GetPointData().SetScalars(vtk_colors)

        glyph = vtk.vtkVertexGlyphFilter()
        glyph.SetInputData(polydata)
        glyph.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(3)

        prop = actor.GetProperty()
        prop.SetRepresentationToPoints()

        self._actors[filename] = actor
        self.renderer.AddActor(self._actors[filename])

        # edl
        basicPasses = vtk.vtkRenderStepsPass()
        self.edl = vtk.vtkEDLShading()
        self.edl.SetDelegatePass(basicPasses)
        glrenderer = vtk.vtkOpenGLRenderer.SafeDownCast(self.renderer)
        glrenderer.SetPass(self.edl)

        self.vtk_widget.GetRenderWindow().Render()

    def reset_camera(self):
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
