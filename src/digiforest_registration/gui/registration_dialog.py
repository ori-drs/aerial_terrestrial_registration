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
)
from PyQt5.QtCore import Qt


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
        self.file_edit = QLineEdit(self.args.uav_cloud if self.args.uav_cloud else "")
        file_browse = QPushButton("Browse...")
        file_browse.clicked.connect(self.browse_file)

        grid.addWidget(file_label, 0, 0)
        grid.addWidget(self.file_edit, 0, 1)
        grid.addWidget(file_browse, 0, 2)

        # --- Row 2: Folder selector ---
        folder_label = QLabel("Select MLS Point Cloud folder:")
        self.folder_edit = QLineEdit(
            self.args.mls_cloud_folder if self.args.mls_cloud_folder else ""
        )
        folder_browse = QPushButton("Browse...")
        folder_browse.clicked.connect(self.browse_folder)

        grid.addWidget(folder_label, 1, 0)
        grid.addWidget(self.folder_edit, 1, 1)
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
        dlg = GlobalShiftScaleDialog(self.args)
        return_value = dlg.exec_()

        if return_value == QDialog.Rejected:
            return self.reject()
        super().accept()

    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if path:
            self.file_edit.setText(path)

    def browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            self.folder_edit.setText(path)


class GlobalShiftScaleDialog(QDialog):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.setWindowTitle("Global shift/scale")
        self.setMinimumWidth(600)

        main_layout = QVBoxLayout()

        # --- Top warning section ---
        warn_label = QLabel(
            "<b>Coordinates are too big (original precision may be lost)!</b>"
        )
        warn_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(warn_label)

        question_label = QLabel("Do you wish to translate/rescale the entity?")
        question_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(question_label)

        info_label = QLabel(
            '<font color="blue">shift/scale information is stored and used '
            "to restore the original coordinates at export time</font>"
        )
        info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(info_label)

        # --- Middle layout ---
        mid_layout = QHBoxLayout()

        # Left box
        left_box = QGroupBox("Point in original coordinate system (on disk)")
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("x = 399094.446998"))
        left_layout.addWidget(QLabel("y = 6786179.336881"))
        left_layout.addWidget(QLabel("z = 161.459090"))
        left_box.setLayout(left_layout)

        # Center shift/scale section
        center_layout = QGridLayout()
        center_layout.addWidget(QLabel("Suggested"), 0, 1)

        center_layout.addWidget(QLabel("+ Shift"), 1, 0)
        center_layout.addWidget(QLineEdit("-399000.00"), 1, 1)
        center_layout.addWidget(QLineEdit("-6786100.00"), 2, 1)
        center_layout.addWidget(QLineEdit("0.00"), 3, 1)

        center_layout.addWidget(QLabel("x Scale"), 4, 0)
        center_layout.addWidget(QLineEdit("1.00000000"), 4, 1)

        # Right box
        right_box = QGroupBox("Point in local coordinate system")
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("x = 94.44700"))
        right_layout.addWidget(QLabel("y = 79.33688"))
        right_layout.addWidget(QLabel("z = 161.45909"))
        right_box.setLayout(right_layout)

        mid_layout.addWidget(left_box)
        mid_layout.addLayout(center_layout)
        mid_layout.addWidget(right_box)

        main_layout.addLayout(mid_layout)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        no_btn = QPushButton("No")
        yes_all_btn = QPushButton("Yes to All")
        yes_btn = QPushButton("Yes")
        btn_layout.addWidget(no_btn)
        btn_layout.addWidget(yes_all_btn)
        btn_layout.addWidget(yes_btn)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)
