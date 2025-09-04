from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QPixmap


class ImageWidget(QWidget):
    def __init__(self):
        super().__init__()

    def load_image(self, image_path):

        # Load image into QPixmap
        pixmap = QPixmap(image_path)

        # Put pixmap into QLabel
        label = QLabel()
        label.setPixmap(pixmap)
        label.setScaledContents(True)  # optional: scale to fit label

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)
