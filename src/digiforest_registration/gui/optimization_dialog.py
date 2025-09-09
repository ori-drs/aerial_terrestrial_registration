from PyQt5.QtWidgets import (
    QDialog,
    QLabel,
    QPushButton,
    QLineEdit,
    QVBoxLayout,
    QGridLayout,
    QDialogButtonBox,
    QFileDialog,
    QMessageBox,
)
import os

from digiforest_registration.gui.registration_dialog import (
    read_first_point_from_ply,
    GlobalShiftScaleDialog,
)


class OptimizationFileFolderDialog(QDialog):
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
        self.mls_folder_edit = QLineEdit(
            self.args.mls_cloud_folder if self.args.mls_cloud_folder else ""
        )
        folder_browse = QPushButton("Browse...")
        folder_browse.clicked.connect(self.browse_folder)

        grid.addWidget(QLabel("Select MLS Point Cloud folder:"), 1, 0)
        grid.addWidget(self.mls_folder_edit, 1, 1)
        grid.addWidget(folder_browse, 1, 2)

        # --- Row 3: Registered Folder selector ---
        self.mls_registered_folder_edit = QLineEdit(
            self.args.mls_registered_cloud_folder
            if self.args.mls_registered_cloud_folder
            else ""
        )
        folder_browse = QPushButton("Browse...")
        folder_browse.clicked.connect(self.browse_registered_mls_folder)

        grid.addWidget(QLabel("Select Registered MLS Point Cloud folder:"), 2, 0)
        grid.addWidget(self.mls_registered_folder_edit, 2, 1)
        grid.addWidget(folder_browse, 2, 2)

        # --- Row 4: Offset label ---
        grid.addWidget(QLabel("Offset"), 3, 0)
        grid.addWidget(QLabel(str(self.args.offset)), 3, 1)
        layout.addLayout(grid)

        # --- Row 5: Output folder ---
        self.output_folder_edit = QLineEdit(
            self.args.optimized_cloud_output_folder
            if self.args.optimized_cloud_output_folder
            else ""
        )
        output_folder_browse = QPushButton("Browse...")
        output_folder_browse.clicked.connect(self.browse_output_folder)

        grid.addWidget(QLabel("Select Output folder:"), 4, 0)
        grid.addWidget(self.output_folder_edit, 4, 1)
        grid.addWidget(output_folder_browse, 4, 2)

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

        if not os.path.isdir(self.mls_registered_folder_edit.text()):
            QMessageBox.critical(
                self,
                "Error",
                "Please select a valid registered MLS point cloud folder.",
            )
            return self.reject()

        self.args.uav_cloud = uav_file
        self.args.mls_cloud_folder = self.mls_folder_edit.text()
        self.args.mls_registered_cloud_folder = self.mls_registered_folder_edit.text()

        # check output path
        if not os.path.isdir(self.output_folder_edit.text()):
            QMessageBox.critical(self, "Error", "Please select a valid output folder.")
            return self.reject()

        self.args.optimized_cloud_output_folder = self.output_folder_edit.text()

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
        path, _ = QFileDialog.getOpenFileName(
            self, "Select File", self.uav_file_edit.text(), "Point Cloud Files (*.ply)"
        )
        if path:
            self.uav_file_edit.setText(path)

    def browse_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Folder", self.mls_folder_edit.text()
        )
        if path:
            self.mls_folder_edit.setText(path)

    def browse_output_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Folder", self.output_folder_edit.text()
        )
        if path:
            self.output_folder_edit.setText(path)

    def browse_registered_mls_folder(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Folder", self.mls_registered_folder_edit.text()
        )
        if path:
            self.mls_registered_folder_edit.setText(path)
