from pathlib import Path
import cv2
import logging


class ExperimentLogger:
    def __init__(self, base_dir: str, version: str = None):
        self._root = Path(base_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._version = version if version is not None else self._get_next_version()
        self._log_dir = self._root / f"version_{self._version}"
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("digiforest_registration")
        # self.logger.setLevel(logging.DEBUG)

    def log_image(self, img, name):
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
