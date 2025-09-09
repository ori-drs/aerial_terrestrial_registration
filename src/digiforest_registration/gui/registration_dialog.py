from PyQt5.QtWidgets import (
    QDialog,
    QLabel,
    QPushButton,
    QLineEdit,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QDialogButtonBox,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import Qt
import struct
import os


def read_first_point_from_ply(filename):
    with open(filename, "rb") as f:
        # --- Step 1: read the header until we know 3 properties ---
        header = []
        xyz_dtypes = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line.startswith("property") and len(xyz_dtypes) < 3:
                dtype = line.split()[1]
                xyz_dtypes.append(dtype)
            if line == "end_header":
                break

        # endianess
        format_line = next(l for l in header if l.startswith("format"))
        if "binary_little_endian" in format_line:
            endian = "<"
        elif "binary_big_endian" in format_line:
            endian = ">"
        else:
            raise ValueError("Only binary PLY supported")

        # mapping from PLY types to struct codes
        type_map = {
            "char": "b",
            "uchar": "B",
            "short": "h",
            "ushort": "H",
            "int": "i",
            "uint": "I",
            "float": "f",
            "double": "d",
        }

        # --- Step 2: build struct just for the first 3 props ---
        fmt = endian + "".join(type_map[d] for d in xyz_dtypes)
        size = struct.calcsize(fmt)

        # --- Step 3: read only that many bytes ---
        data = f.read(size)
        x, y, z = struct.unpack(fmt, data)
        return x, y, z


class FileFolderDialog(QDialog):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("Select Inputs")
        self.setMinimumWidth(500)

        layout = QVBoxLayout()

        grid = QGridLayout()

        # --- Row 1: File selector ---
        file_label = QLabel("Select UAV Point Cloud:")
        self.uav_file_edit = QLineEdit(
            self.args.uav_cloud if self.args.uav_cloud else ""
        )
        file_browse = QPushButton("Browse...")
        file_browse.clicked.connect(self.browse_file)

        grid.addWidget(file_label, 0, 0)
        grid.addWidget(self.uav_file_edit, 0, 1)
        grid.addWidget(file_browse, 0, 2)

        # --- Row 2: Folder selector ---
        folder_label = QLabel("Select MLS Point Cloud folder:")
        self.mls_folder_edit = QLineEdit(
            self.args.mls_cloud_folder if self.args.mls_cloud_folder else ""
        )
        folder_browse = QPushButton("Browse...")
        folder_browse.clicked.connect(self.browse_folder)

        grid.addWidget(folder_label, 1, 0)
        grid.addWidget(self.mls_folder_edit, 1, 1)
        grid.addWidget(folder_browse, 1, 2)

        # --- Row 3: Two text labels ---
        grid.addWidget(QLabel("Offset"), 2, 0)
        grid.addWidget(QLabel(str(self.args.offset)), 2, 1)

        layout.addLayout(grid)

        # --- OK / Cancel Buttons ---
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def accept(self):
        # read the first point of the UAV cloud to check if offset is needed
        uav_file = self.uav_file_edit.text()
        if not uav_file or not os.path.isfile(uav_file):
            QMessageBox.critical(
                self, "Error", "Please select a valid UAV point cloud file."
            )
            return self.reject()

        first_point = read_first_point_from_ply(uav_file)

        if not self._is_new_offset_needed(first_point, self.args.offset):
            return super().accept()

        dlg = GlobalShiftScaleDialog(self.args, first_point)
        return_value = dlg.exec_()

        if return_value == QDialog.Rejected:
            return self.reject()

        # read uav and mls paths and check that they exist
        if not os.path.isdir(self.mls_folder_edit.text()):
            QMessageBox.critical(
                self, "Error", "Please select a valid MLS point cloud folder."
            )
            return self.reject()

        self.args.uav_cloud = uav_file
        self.args.mls_cloud_folder = self.mls_folder_edit.text()
        return super().accept()

    def _is_new_offset_needed(self, point, offset):
        if offset is not None:
            max_offset = max(
                point[0] + offset[0], point[1] + offset[1], point[2] + offset[2]
            )
            if max_offset > 1e4:
                return True
            else:
                return False

        return True

    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if path:
            self.uav_file_edit.setText(path)

    def browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            self.mls_folder_edit.setText(path)


class GlobalShiftScaleDialog(QDialog):
    def __init__(self, args, pt):
        super().__init__()
        self.args = args
        self.pt = pt
        self.setWindowTitle("Global shift/scale")
        self.setMinimumWidth(600)

        main_layout = QVBoxLayout()

        # --- Top warning section ---
        warn_label = QLabel(
            "<b>Coordinates are too big (original precision may be lost)!</b>"
        )
        warn_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(warn_label)

        question_label = QLabel("Do you wish to translate the entity?")
        question_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(question_label)

        info_label = QLabel(
            "shift information is stored and used "
            "to restore the original coordinates at export time"
        )
        info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(info_label)

        # --- Middle layout ---
        mid_layout = QHBoxLayout()

        # Left box
        left_box = QGroupBox("Point in original coordinate system (on disk)")
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("x = " + f"{pt[0]:.1f}"))
        left_layout.addWidget(QLabel("y = " + f"{pt[1]:.1f}"))
        left_layout.addWidget(QLabel("z = " + f"{pt[2]:.1f}"))
        left_box.setLayout(left_layout)

        # Center shift section
        center_layout = QGridLayout()
        center_layout.addWidget(QLabel("Offset"), 0, 1)

        center_layout.addWidget(QLabel("+ Shift"), 1, 0)
        self.line_edit_offset_x = QLineEdit(f"{self.args.offset[0]:.1f}")
        self.line_edit_offset_y = QLineEdit(f"{self.args.offset[1]:.1f}")
        self.line_edit_offset_z = QLineEdit(f"{self.args.offset[2]:.1f}")
        self.line_edit_offset_x.editingFinished.connect(self.offset_updated)
        self.line_edit_offset_y.editingFinished.connect(self.offset_updated)
        self.line_edit_offset_z.editingFinished.connect(self.offset_updated)
        center_layout.addWidget(self.line_edit_offset_x, 1, 1)
        center_layout.addWidget(self.line_edit_offset_y, 2, 1)
        center_layout.addWidget(self.line_edit_offset_z, 3, 1)

        # Right box
        right_box = QGroupBox("Point in local coordinate system")
        right_layout = QVBoxLayout()
        self.label_updated_x = QLabel("x = " + f"{pt[0]+self.args.offset[0]:.1f}")
        self.label_updated_y = QLabel("x = " + f"{pt[1]+self.args.offset[1]:.1f}")
        self.label_updated_z = QLabel("x = " + f"{pt[2]+self.args.offset[2]:.1f}")
        right_layout.addWidget(self.label_updated_x)
        right_layout.addWidget(self.label_updated_y)
        right_layout.addWidget(self.label_updated_z)
        right_box.setLayout(right_layout)

        mid_layout.addWidget(left_box)
        mid_layout.addLayout(center_layout)
        mid_layout.addWidget(right_box)

        main_layout.addLayout(mid_layout)

        # --- Buttons ---
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        self.setLayout(main_layout)

    def offset_updated(self):
        # saving offset
        self.args.offset[0] = float(self.line_edit_offset_x.text())
        self.args.offset[1] = float(self.line_edit_offset_y.text())
        self.args.offset[2] = float(self.line_edit_offset_z.text())

        self.label_updated_x.setText(f"{self.pt[0]+self.args.offset[0]:.1f}")
        self.label_updated_y.setText(f"{self.pt[1]+self.args.offset[1]:.1f}")
        self.label_updated_z.setText(f"{self.pt[2]+self.args.offset[2]:.1f}")
