import os
from argparse import ArgumentParser

from .settings import CURSOR_DATASET, MODEL_DIR, SCALER_FILE

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")

    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("-o", "--output", default=CURSOR_DATASET)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("-d", "--dataset", default=CURSOR_DATASET)
    train_parser.add_argument("-m", "--model-dir", default=MODEL_DIR)
    train_parser.add_argument("-s", "--scaler-file", default=SCALER_FILE)
    train_parser.add_argument("-e", "--epochs", default=100)

    simulate_parser = subparsers.add_parser("simulate")
    simulate_parser.add_argument("-m", "--model-dir", default=MODEL_DIR)
    simulate_parser.add_argument("-s", "--scaler-file", default=SCALER_FILE)
    simulate_parser.add_argument("-i", "--iterations", default=-1)

    args = parser.parse_args()

    match args.action:
        case "record":
            from .record import record

            record(args.output)
        case "train":
            from .train import train

            train(args.dataset, args.model_dir, args.scaler_file, args.epochs)
        case "simulate":
            from .simulate import simulate

            simulate(args.model_dir, args.scaler_file, args.iterations)


if __name__ == "__main__":
    main()
