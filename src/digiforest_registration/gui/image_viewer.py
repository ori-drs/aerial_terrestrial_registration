from PyQt5.QtGui import QPixmap

from PyQt5.QtCore import Qt
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
    def __init__(self, pixmap: QPixmap, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self._pixitem = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixitem)
        self.setScene(self._scene)

        # Better zoom/pan UX
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)

        self._min_scale = 0.05
        self._max_scale = 50.0
        self._current_scale = 1.0

        self.fitInView(self._pixitem, Qt.KeepAspectRatio)
        self._current_scale = self._extract_scale()

    def wheelEvent(self, event):
        # Ctrl+wheel to zoom; plain wheel scrolls
        if not (event.modifiers() & Qt.ControlModifier):
            return super().wheelEvent(event)

        angle = event.angleDelta().y()
        step = 1.0015**angle  # smooth zoom
        self._apply_scale(step)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):  # '+' or '='
            self._apply_scale(1.1)
        elif event.key() == Qt.Key_Minus:
            self._apply_scale(1 / 1.1)
        elif event.key() in (Qt.Key_0,):  # reset zoom (Ctrl+0 common)
            self.reset_zoom()
        elif event.key() == Qt.Key_F:
            self.fit_to_window()
        else:
            super().keyPressEvent(event)

    def reset_zoom(self):
        self.setTransform(self.transform().fromScale(1.0, 1.0).identity())
        self._current_scale = 1.0
        self.centerOn(self._pixitem)

    def fit_to_window(self):
        if self._pixitem.pixmap().isNull():
            return
        self.fitInView(self._pixitem, Qt.KeepAspectRatio)
        self._current_scale = self._extract_scale()

    def _apply_scale(self, factor):
        new_scale = self._current_scale * factor
        if new_scale < self._min_scale:
            factor = self._min_scale / self._current_scale
            new_scale = self._min_scale
        elif new_scale > self._max_scale:
            factor = self._max_scale / self._current_scale
            new_scale = self._max_scale

        self.scale(factor, factor)
        self._current_scale = new_scale

    def _extract_scale(self):
        # Get uniform scale from the view transform
        m = self.transform()
        # sqrt of (m11*m22 - m12*m21) would be exact; for uniform scaling m11 is enough.
        return m.m11()


class ImageWidget(QWidget):
    def __init__(self):
        super().__init__()

    def load_image(self, image_path: str):
        pm = QPixmap(image_path)
        self.view = ImageView(pm)

        # Controls
        zoom_in_btn = QPushButton("+")
        zoom_out_btn = QPushButton("–")
        fit_btn = QPushButton("Fit")

        zoom_in_btn.clicked.connect(lambda: self.view._apply_scale(1.1))
        zoom_out_btn.clicked.connect(lambda: self.view._apply_scale(1 / 1.1))
        fit_btn.clicked.connect(self.view.fit_to_window)

        controls = QHBoxLayout()
        controls.addWidget(zoom_out_btn)
        controls.addWidget(zoom_in_btn)
        controls.addWidget(fit_btn)
        controls.addStretch()

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.view)

        self.view.fit_to_window()
