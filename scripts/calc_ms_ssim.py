import argparse
from src.evaluation.mssim import calc_ms_ssim_for_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Path to dataset")
    args = parser.parse_args()

    mean, sd = calc_ms_ssim_for_path(args.dataset, n=100, trials=1)


if __name__ == "__main__":
    main()