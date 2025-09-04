from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPixmap, QPainter, QTransform
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
)


class ImageView(QGraphicsView):
    """Zoomable/pannable image view. Starts empty; call set_pixmap() later."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixitem = QGraphicsPixmapItem()  # empty initially
        self._scene.addItem(self._pixitem)
        self.setScene(self._scene)

        # Nice rendering + interactions
        self.setRenderHints(
            self.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)

        self._min_scale = 0.05
        self._max_scale = 50.0
        self._current_scale = 1.0

    def has_image(self) -> bool:
        return not self._pixitem.pixmap().isNull()

    def clear_pixmap(self, reset: bool = True):
        """Remove the image but keep the view and scene intact."""
        if not self.has_image():
            return
        self._pixitem.setPixmap(QPixmap())  # empty pixmap
        self._scene.setSceneRect(QRectF())  # no content bounds
        if reset:
            self.setTransform(QTransform())  # identity
            self._current_scale = 1.0

    def set_pixmap(self, pm: QPixmap, reset: bool = True):
        """Replace the displayed image. If reset=True, fit and reset zoom."""
        if pm.isNull():
            return
        self._pixitem.setPixmap(pm)
        self._scene.setSceneRect(self._pixitem.boundingRect())
        if reset:
            self.fit_to_window()
            self._current_scale = self._extract_scale()

    def reset_zoom(self):
        if self._pixitem.pixmap().isNull():
            return
        self.setTransform(self.transform().fromScale(1.0, 1.0).identity())
        self._current_scale = 1.0
        self.centerOn(self._pixitem)

    def fit_to_window(self):
        if self._pixitem.pixmap().isNull():
            return
        self.fitInView(self._pixitem, Qt.KeepAspectRatio)
        self._current_scale = self._extract_scale()

    # --- interactions ---
    def wheelEvent(self, event):
        # Ctrl+wheel to zoom; otherwise let the view scroll normally
        if self._pixitem.pixmap().isNull():
            return
        if not (event.modifiers() & Qt.ControlModifier):
            return super().wheelEvent(event)
        angle = event.angleDelta().y()
        step = 1.0015**angle  # smooth zoom
        self._apply_scale(step)

    def keyPressEvent(self, event):
        if self._pixitem.pixmap().isNull():
            return super().keyPressEvent(event)
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self._apply_scale(1.1)
        elif event.key() == Qt.Key_Minus:
            self._apply_scale(1 / 1.1)
        elif event.key() == Qt.Key_0:
            self.reset_zoom()
        elif event.key() == Qt.Key_F:
            self.fit_to_window()
        else:
            super().keyPressEvent(event)

    # --- internals ---
    def _apply_scale(self, factor: float):
        new_scale = self._current_scale * factor
        if new_scale < self._min_scale:
            factor = self._min_scale / self._current_scale
            new_scale = self._min_scale
        elif new_scale > self._max_scale:
            factor = self._max_scale / self._current_scale
            new_scale = self._max_scale
        self.scale(factor, factor)
        self._current_scale = new_scale

    def _extract_scale(self) -> float:
        m = self.transform()
        return m.m11()  # uniform scale


class ImageWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.view = ImageView()

        # Controls
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_out = QPushButton("–")
        self.btn_fit = QPushButton("Fit")

        self.btn_zoom_in.clicked.connect(lambda: self.view._apply_scale(1.1))
        self.btn_zoom_out.clicked.connect(lambda: self.view._apply_scale(1 / 1.1))
        self.btn_fit.clicked.connect(self.view.fit_to_window)

        # Start disabled until an image is loaded
        self._set_controls_enabled(False)

        controls = QHBoxLayout()
        for w in (self.btn_zoom_out, self.btn_zoom_in, self.btn_fit):
            controls.addWidget(w)
        controls.addStretch()

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.view)

    def _set_controls_enabled(self, enabled: bool):
        for b in (self.btn_zoom_in, self.btn_zoom_out, self.btn_fit):
            b.setEnabled(enabled)

    def delete_image(self):
        """Remove the displayed image and disable zoom/fit controls."""
        self.view.clear_pixmap(reset=True)
        self._set_controls_enabled(False)

    def load_image(self, image_path: str, reset: bool = True):
        pm = QPixmap(image_path)
        if pm.isNull():
            return
        self.view.set_pixmap(pm, reset=reset)
        # enable controls now that we have an image
        for b in (self.btn_zoom_in, self.btn_zoom_out, self.btn_fit):
            b.setEnabled(True)
