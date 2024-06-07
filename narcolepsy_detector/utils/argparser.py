import os
from argparse import ArgumentParser


def none_or_str(value):
    if value == "None":
        return None

    return value


def eval_arguments():

    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_dir",
        default="data/narco_features/avg_kw21_long_test_unscaled.csv",
        type=str,
        help="Path to supplied data directory",
        nargs="+",
    )
    parser.add_argument(
        "-e",
        "--experiment_dir",
        required=True,
        type=str,
        help="Path to directory containing trained model.",
        nargs="+",
    )
    parser.add_argument(
        "-s",
        "--savedir_output",
        default=os.path.join("outputs", "roc_ensemble_default.png"),
        type=str,
        help="Path to save output ROC curves.",
    )
    parser.add_argument("--data-master", type=none_or_str)

    return parser.parse_args()


def train_arguments():

    parser = ArgumentParser()
    parser.add_argument("-c", "--kernel", type=str, default="rbf")
    parser.add_argument("-d", "--data_dir", default="data/narco_features", type=str)
    parser.add_argument("-f", "--feature_set", default="all", type=str)
    parser.add_argument("-i", "--inducing", type=str, default="data", choices=["data", "kmeans"])
    parser.add_argument("-k", "--n_kfold", default=5, type=int)
    parser.add_argument(
        "-l",
        "--likelihood",
        type=str,
        default="gaussian",
        choices=["bernoulli", "gaussian"],
    )
    parser.add_argument("-m", "--model", required=True, type=str)
    parser.add_argument("-n", "--n_repeats", default=1, type=int)
    parser.add_argument("--repeat-idx", type=int, default=None)
    parser.add_argument("-p", "--n_inducing_points", default=350, type=int)
    parser.add_argument("-r", "--resolution", type=float)
    parser.add_argument("-s", "--save_dir", type=str)
    parser.add_argument("--baseline-model", action="store_true", default=False)
    parser.add_argument("--data-master", type=str, default="data_master.csv")
    parser.add_argument("--gp_model", type=str, default="svgp")
    parser.add_argument("--log_frequency", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=10000)
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--pca_components", default=100)
    parser.add_argument("--pca_explained", default=90)
    parser.add_argument("--feature_selection", action="store_true", default=False)
    parser.add_argument("--feature_select_method", default="f1")
    parser.add_argument("--feature_select_specificity", default=0.90)
    parser.add_argument("--feature_select_prevalence", default=0.01)
    parser.add_argument("--smote", dest="smote", action="store_true")
    parser.add_argument("--no-smote", dest="smote", action="store_false")
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="Run model optimization without STAGES and without sleep scoring eval data.",
    )
    parser.add_argument(
        "--no-stages", action="store_true", default=False, help="Run model optimization without STAGES data."
    )
    parser.set_defaults(smote=True)

    return parser


def adjust_args(args):

    # HACK: Parse the resolution argument
    args.resolution = int(args.resolution) if args.resolution % 1 == 0 else args.resolution

    # Parse experiment dir
    if not args.save_dir:
        if args.resolution:
            args.save_dir = os.path.join(
                "experiments",
                f"{args.model}_r{args.resolution:02}_{args.kernel}_{args.feature_set}_k{args.n_kfold:02}_n{args.n_repeats:02}_{args.inducing}",
            )
        else:
            args.save_dir = os.path.join(
                "experiments",
                f"{args.model}_{args.kernel}_{args.feature_set}_k{args.n_kfold:02}_n{args.n_repeats:02}_{args.inducing}",
            )
    return args
