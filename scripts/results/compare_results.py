import numpy as np
import argparse
from digiforest_registration.utils import rotation_matrix_to_euler


def parse_inputs():
    parser = argparse.ArgumentParser(
        prog="cloud_registration",
        description="Registers a frontier cloud to a reference UAV cloud",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("--manual_file", default=None, help="")
    parser.add_argument("--auto_file", default=None, help="")
    args = parser.parse_args()

    return args


def unpack_matrix(mat: np.ndarray):
    return (
        str(mat[0, 0])
        + " "
        + str(mat[0, 1])
        + " "
        + str(mat[0, 2])
        + " "
        + str(mat[0, 3])
        + " "
        + str(mat[1, 0])
        + " "
        + str(mat[1, 1])
        + " "
        + str(mat[1, 2])
        + " "
        + str(mat[1, 3])
        + " "
        + str(mat[2, 0])
        + " "
        + str(mat[2, 1])
        + " "
        + str(mat[2, 2])
        + " "
        + str(mat[2, 3])
        + " "
        + str(mat[3, 0])
        + " "
        + str(mat[3, 1])
        + " "
        + str(mat[3, 2])
        + " "
        + str(mat[3, 3])
    )


def parse_registration(registration_file: str, results: dict):
    with open(registration_file, "r") as file:
        lines = file.readlines()
        for line in lines:
            tokens = line.strip().split(" ")

            # Parse lines
            if tokens[0] == "#":
                continue

            t = np.array(
                [
                    [
                        float(tokens[1]),
                        float(tokens[2]),
                        float(tokens[3]),
                        float(tokens[4]),
                    ],
                    [
                        float(tokens[5]),
                        float(tokens[6]),
                        float(tokens[7]),
                        float(tokens[8]),
                    ],
                    [
                        float(tokens[9]),
                        float(tokens[10]),
                        float(tokens[11]),
                        float(tokens[12]),
                    ],
                    [
                        float(tokens[13]),
                        float(tokens[14]),
                        float(tokens[15]),
                        float(tokens[16]),
                    ],
                ]
            )
            mean_rmse = float(tokens[17])
            std_rmse = float(tokens[18])
            results[tokens[0]] = {
                "transform": t,
                "mean_rmse": mean_rmse,
                "std_rmse": std_rmse,
            }


def compare_transform(t1: np.ndarray, t2: np.ndarray):
    # Compute the difference between the two transforms
    diff = np.linalg.norm(t1[0:3, 3] - t2[0:3, 3])
    euler_angles_1 = rotation_matrix_to_euler(t1[0:3, 0:3])
    euler_angles_2 = rotation_matrix_to_euler(t2[0:3, 0:3])
    return diff, np.linalg.norm(euler_angles_1 - euler_angles_2)


if __name__ == "__main__":
    manual_registration = {}
    auto_registration = {}

    args = parse_inputs()

    parse_registration(args.manual_file, manual_registration)
    parse_registration(args.auto_file, auto_registration)

    mean_rmse_manual = []
    mean_rmse_auto = []
    std_rmse_manual = []
    std_rmse_auto = []
    for file, data in manual_registration.items():
        t_manual = manual_registration[file]["transform"]
        t_auto = auto_registration[file]["transform"]
        t_diff, r_diff = compare_transform(t_manual, t_auto)

        mean_rmse_manual.append(manual_registration[file]["mean_rmse"])
        mean_rmse_auto.append(auto_registration[file]["mean_rmse"])

        std_rmse_manual.append(manual_registration[file]["std_rmse"])
        std_rmse_auto.append(auto_registration[file]["std_rmse"])

        # print(
        #     f"File: {file}, translation error {t_diff}, rotation error {r_diff}, mean rmse error {mean_rmse_manual - mean_rmse_auto}, std rmse error {std_rmse_manual - std_rmse_auto}"
        # )
    print(
        "Mean manual :", np.mean(mean_rmse_manual), "mean auto", np.mean(mean_rmse_auto)
    )
    print("Std manual :", np.mean(std_rmse_manual), "std auto", np.mean(std_rmse_auto))
