import random
import numpy as np
import open3d as o3d
from vtk.util import numpy_support
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
        self.renderer.AutomaticLightCreationOn()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.interactor.Initialize()

        self._actor = None
        self._setup_scene()

    def _setup_scene(self):
        self.renderer.SetBackground(1.0, 1.0, 1.0)
        # self.add_demo_points()
        self.load_point_cloud(None)
        self.renderer.ResetCamera()

    def load_point_cloud(self, filename):
        cloud = o3d.t.io.read_point_cloud(
            "/home/benoit/code/digiforest_drs/digiforest_registration/final_registration.ply"
        )
        xyz = cloud.point["positions"].numpy()
        rgb = 255 * cloud.point["colors"].numpy()

        rgb = rgb.astype(np.uint8)
        points = vtk.vtkPoints()
        # points.SetData(numpy_support.numpy_to_vtk(xyz.astype(np.float32)))
        for i in range(xyz.shape[0]):
            points.InsertNextPoint(xyz[i, 0], xyz[i, 1], xyz[i, 2])

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)

        vtk_colors = numpy_support.numpy_to_vtk(
            rgb, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
        )
        vtk_colors.SetName("Colors")
        polydata.GetPointData().SetScalars(vtk_colors)

        # vtk_normals = numpy_support.numpy_to_vtk(normals.astype(np.float32))
        # vtk_normals.SetNumberOfComponents(3)
        # vtk_normals.SetName("Normals")
        # polydata.GetPointData().SetNormals(vtk_normals)

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
        # prop.LightingOff() # otherwise point cloud is black when normals are defined

        if self._actor:
            self.renderer.RemoveActor(self._actor)
        self._actor = actor
        self.renderer.AddActor(actor)

        # edl
        basicPasses = vtk.vtkRenderStepsPass()
        edl = vtk.vtkEDLShading()
        edl.SetDelegatePass(basicPasses)
        glrenderer = vtk.vtkOpenGLRenderer.SafeDownCast(self.renderer)
        glrenderer.SetPass(edl)

        self.vtk_widget.GetRenderWindow().Render()

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

        rgb = (np.random.rand(n, 3) * 255).astype(np.uint8)
        vtk_colors = numpy_support.numpy_to_vtk(
            rgb, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
        )
        vtk_colors.SetName("Colors")
        poly.GetPointData().SetScalars(vtk_colors)

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
