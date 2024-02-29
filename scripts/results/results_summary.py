import argparse
import pickle


def parse_inputs():
    parser = argparse.ArgumentParser(
        prog="results_summary",
        description="Results summary",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("--results_file", default=None, help="")
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_inputs()

    file = open(args.results_file, "rb")
    data = pickle.load(file)
    file.close()

    num_success = 0
    for file, result in data.items():
        print(
            f"File {file}, {result.transform}, {result.icp_fitness}, {result.success}"
        )
        if result.success:
            num_success += 1

    print(
        f"Number of successes : {num_success}, number of elements {len(data)}, ratio {float(num_success)/len(data)}"
    )
