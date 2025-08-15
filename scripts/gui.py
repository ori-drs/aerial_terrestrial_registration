import sys

from PyQt5.QtWidgets import QApplication
from digiforest_registration.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
