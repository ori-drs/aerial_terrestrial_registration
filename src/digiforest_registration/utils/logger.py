from pathlib import Path
import shutil
import cv2
import logging
import open3d as o3d


class ExperimentLogger:
    def __init__(
        self, base_dir: str, version: str = None, log_pointclouds: bool = False
    ):
        self.log_pointclouds = log_pointclouds
        self._root = Path(base_dir)
        self._version = version if version is not None else self._get_next_version()
        self._root = Path(base_dir) / f"version_{self._version}"
        self._root.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("digiforest_registration")

    def current_logging_directory(self) -> str:
        return str(self._root)

    def set_leaf_logging_folder(self, name: str):
        self._log_dir = self._root / name
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def log_image(self, img, name: str):
        """Save image in log folder
        If an image with the same name already exists, it will create
        a new unique name for the image

        """
        img_path = self._log_dir / f"{name}.png"
        if img_path.exists():
            # get new unique name
            version = 1
            while img_path.exists():
                img_path = self._log_dir / f"{name}_{version}.png"
                version += 1
        cv2.imwrite(str(img_path), img)

    def log_pointcloud(self, cloud, name: str):
        """Save pointcloud in log folder"""
        if not self.log_pointclouds:
            return
        path = self._log_dir / f"{name}.ply"
        o3d.t.io.write_point_cloud(str(path), cloud)

    def delete_all_logs(self):
        if hasattr(self, "_root"):
            shutil.rmtree(self._root)

    def _get_next_version(self) -> int:
        if not self._root.is_dir():
            return 0

        existing_versions = []
        for d in self._root.iterdir():
            name = d.name
            if d.is_dir() and name.startswith("version_"):
                dir_ver = name.split("_")[1]
                if dir_ver.isdigit():
                    existing_versions.append(int(dir_ver))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1

    def info(self, data):
        self.logger.info(data)

    def debug(self, data):
        self.logger.debug(data)

    def warning(self, data):
        self.logger.warning(data)

    def error(self, data):
        self.logger.error(data)
