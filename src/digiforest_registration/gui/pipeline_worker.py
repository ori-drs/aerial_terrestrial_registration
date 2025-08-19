import numpy as np
import os

from PyQt5.QtCore import QObject, pyqtSignal
from digiforest_registration.utils import (
    check_registration_inputs_validity,
    CloudIO,
)
from digiforest_registration.utils import crop_cloud, crop_cloud_to_size
from digiforest_registration.registration.registration import Registration


class PipelineWorker(QObject):
    finished = pyqtSignal(bytes)
    progress = pyqtSignal(str)

    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self):
        args = self.args
        try:
            print("Initialization...")
            # Check validity of inputs
            (
                mls_cloud_filenames,
                mls_cloud_folder,
                uav_cloud_filename,
            ) = check_registration_inputs_validity(args)

            # Loading the data
            if args.offset is not None and len(args.offset) == 3:
                offset = np.array(
                    [
                        args.offset[0],
                        args.offset[1],
                        args.offset[2],
                    ],
                    dtype=np.float32,
                )
            else:
                # default offset
                offset = np.array([0, 0, 0], dtype=np.float32)

            cloud_io = CloudIO(
                offset, logger=None, downsample_cloud=args.downsample_cloud
            )
            uav_cloud = cloud_io.load_cloud(str(uav_cloud_filename))

            # if (
            #     args.mls_registered_cloud_folder is not None
            #     and args.tiles_conf_file is not None
            # ):
            #     tile_config_reader = TileConfigReader(
            #         args.tiles_conf_file, offset
            #     )

            for mls_cloud_filename in mls_cloud_filenames:

                original_mls_cloud = cloud_io.load_cloud(str(mls_cloud_filename))

                # cropping input clouds
                if args.crop_mls_cloud:
                    mls_cloud = crop_cloud_to_size(original_mls_cloud, size=30)
                else:
                    mls_cloud = original_mls_cloud
                cropped_uav_cloud = crop_cloud(uav_cloud, mls_cloud, padding=20)

                logging_dir = args.logging_dir
                if args.logging_dir is None:
                    logging_dir = "./logs"
                logging_dir = os.path.join(logging_dir, mls_cloud_filename.stem)

                registration = Registration(
                    cropped_uav_cloud,
                    mls_cloud,
                    args.ground_segmentation_method,
                    args.correspondence_matching_method,
                    args.mls_feature_extraction_method,
                    args.icp_fitness_score_threshold,
                    args.min_distance_between_peaks,
                    args.max_number_of_clique,
                    logging_dir,
                    correspondence_graph_distance_threshold=args.correspondence_graph_distance_threshold,
                    maximum_rotation_offset=args.maximum_rotation_offset,
                    debug=args.debug,
                )
                registration = registration
                print("Starting registration")
                # success = registration.registration()

        except Exception as e:
            print(f"Registration failed: {e}")
            pass
            # self.failed.emit(str(e))
