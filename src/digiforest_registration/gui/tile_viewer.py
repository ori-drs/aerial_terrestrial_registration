from PyQt5 import QtWidgets
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QRectF, QPointF


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
