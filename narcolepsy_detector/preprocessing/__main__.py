import sys
from argparse import ArgumentParser

from narcolepsy_detector.preprocessing import prepare_data, merge_dfs
from narcolepsy_detector.utils import get_logger

logger = get_logger()


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
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--save_file", default=None, type=str)
    parser.add_argument("--data-master", type=none_or_str, default="data_files/data-overview-stanford-takeda.csv")
    args = parser.parse_args()

    logger.info(f'Usage: {" ".join([x for x in sys.argv])}\n')
    logger.info("Settings:")
    logger.info("---------")
    for idx, (k, v) in enumerate(sorted(vars(args).items())):
        if idx == len(vars(args)) - 1:
            logger.info(f"{k:>15}\t{v}\n")
        else:
            logger.info(f"{k:>15}\t{v}")

    if args.merge:
        merge_dfs(args.output_dir)
    else:
        prepare_data(args.data_dir, args.resolution, args.output_dir, args.subset, args.save_file, args.data_master)
