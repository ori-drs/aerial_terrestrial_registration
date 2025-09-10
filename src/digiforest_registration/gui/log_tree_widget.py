from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTreeView
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QStandardItemModel, QStandardItem
import os


class FileTreeWidget(QWidget):
    fileChecked = pyqtSignal(str)  # signal that emits the filename when checked
    fileUnChecked = pyqtSignal(str)

    def __init__(self, root_path: str, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        self.tree = QTreeView()
        layout.addWidget(self.tree)

        # model for holding items
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(["Files"])
        self.tree.setModel(self.model)
        self.tree.setHeaderHidden(False)

        # populate tree
        self.root_path = root_path
        self.additional_filepaths = []
        self.update_view()

        # connect
        self.model.itemChanged.connect(self.on_item_changed)

    def add_file(self, filepath: str):
        if filepath not in self.additional_filepaths:
            self.additional_filepaths.append(filepath)
            self.update_view()

    def update_root_path(self, path: str):
        self.root_path = path
        self.update_view()

    def update_view(self):
        self.clear_tree()
        if self.root_path and os.path.isdir(self.root_path):
            self.populate_tree(self.root_path, self.model.invisibleRootItem())

        for filepath in self.additional_filepaths:
            if not filepath:
                continue
            item = QStandardItem(os.path.basename(filepath))
            item.setData(filepath, Qt.UserRole)

            item.setFlags(
                Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable
            )
            item.setCheckState(Qt.Unchecked)
            self.model.invisibleRootItem().appendRow(item)

    def clear_tree(self):
        self.model.removeRows(0, self.model.rowCount())

    def populate_tree(self, path, parent_item):
        """Recursively add folders and files to the tree."""
        for entry in sorted(os.listdir(path)):
            full_path = os.path.join(path, entry)

            # ignore items that are not ply/png or folder
            if not (
                entry.endswith(".ply")
                or entry.endswith(".png")
                or os.path.isdir(full_path)
            ):
                continue

            item = QStandardItem(entry)
            item.setData(full_path, Qt.UserRole)

            if os.path.isdir(full_path):
                item.setFlags(Qt.ItemIsEnabled)  # folder not checkable
                parent_item.appendRow(item)
                self.populate_tree(full_path, item)
            else:
                item.setFlags(
                    Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable
                )
                item.setCheckState(Qt.Unchecked)
                parent_item.appendRow(item)

    def on_item_changed(self, item):
        """Emit signal if a file item is checked."""
        full_path = item.data(Qt.UserRole)
        if os.path.isfile(full_path) and item.checkState() == Qt.Checked:
            self.fileChecked.emit(full_path)
        elif os.path.isfile(full_path) and item.checkState() == Qt.Unchecked:
            self.fileUnChecked.emit(full_path)

    def uncheck_all_except(self, item):
        """Uncheck all items in the tree except the given item."""
        root = self.model.invisibleRootItem()
        for i in range(root.rowCount()):
            self.uncheck_item(root.child(i), item)

    def uncheck_item(self, item, except_item):
        """Uncheck the given item and its children, except for the specified item."""
        if item != except_item:
            item.setCheckState(Qt.Unchecked)
        for i in range(item.rowCount()):
            self.uncheck_item(item.child(i), except_item)
