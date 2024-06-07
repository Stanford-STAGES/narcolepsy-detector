import sys
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from narcolepsy_detector.preprocessing.__main__ import prepare_data
from narcolepsy_detector.utils import get_logger


logger = get_logger()


def select_files_from_datamaster(current_split, n_splits, datamaster):

    df = pd.read_csv(datamaster)
    df_split = np.array_split(df, n_splits)

    return df_split[current_split - 1]


if __name__ == "__main__":

    def none_or_str(value):
        if value == "None":
            return None
        return value

    parser = ArgumentParser()
    parser.add_argument("-d", "--data-dir", type=str, default="./data/massc")
    parser.add_argument("-o", "--output-dir", type=str, default="./data/narco_features")
    parser.add_argument("-r", "--resolution", type=float, nargs="+", default=15)
    parser.add_argument("-c", "--cohort", type=none_or_str, default=None)
    parser.add_argument("-s", "--subset", type=str)
    parser.add_argument("--save-file", default=None, type=str)
    parser.add_argument("--data-master", type=none_or_str, default="data_files/data-overview-stanford-takeda.csv")
    parser.add_argument("--current-split", type=int, default=1)
    parser.add_argument("--splits", default=1, type=int)
    parser.add_argument('-f', "--features", type=str, nargs="+", default='standard', choices=['standard', 'mtm-scored', 'mtm-argmax', 'mtm-scored+cli'])
    args = parser.parse_args()

    logger.info(f'Usage: {" ".join([x for x in sys.argv])}\n')
    logger.info("Settings:")
    logger.info("---------")
    for idx, (k, v) in enumerate(sorted(vars(args).items())):
        if idx == len(vars(args)) - 1:
            logger.info(f"{k:>15}\t{v}\n")
        else:
            logger.info(f"{k:>15}\t{v}")

    datamaster = select_files_from_datamaster(args.current_split, args.splits, args.data_master)

    save_file = f'r{args.resolution[0]:.2f}_unscaled_{args.current_split-1:03}.csv'

    prepare_data(args.data_dir, args.resolution, args.output_dir, args.subset, save_file, datamaster, args.features)
