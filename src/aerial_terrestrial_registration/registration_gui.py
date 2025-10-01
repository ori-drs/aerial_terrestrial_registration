import sys

from PyQt5.QtWidgets import QApplication
from aerial_terrestrial_registration.gui.main_window import MainWindow
from aerial_terrestrial_registration.utils import parse_inputs


def main():
    app = QApplication(sys.argv)
    args = parse_inputs()
    win = MainWindow(args)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
