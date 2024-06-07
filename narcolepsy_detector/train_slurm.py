import numpy as np

from narcolepsy_detector.train_single_model import train_single_model
from narcolepsy_detector.utils.argparser import train_arguments, adjust_args
from narcolepsy_detector.utils.logger import get_logger


logger = get_logger()


if __name__ == "__main__":

    parser = train_arguments()
    parser.add_argument('--current-fold', type=int, default=1)
    parser.add_argument('--current-repeat', type=int, default=1)
    args = parser.parse_args()

    # Check if current fold is smaller than total number of splits
    if args.current_fold <= args.n_kfold and args.current_repeat <= args.n_repeats:

        # Set seed
        np.random.seed(42)

        args = adjust_args(args)

        train_single_model(**vars(args), options=args)

    else:
        logger.info(f'Received fold idx ({args.current_fold}) is larger than the total number of folds ({args.n_kfold})!')
