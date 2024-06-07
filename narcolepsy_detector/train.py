from argparse import ArgumentParser

import numpy as np

from narcolepsy_detector.data.data_class import SingleResolutionDataset
from narcolepsy_detector.model import NarcolepsyModel
from narcolepsy_detector.utils import get_logger


logger = get_logger()


def train_model(
    model,
    resolution,
    feature_set,
    n_kfold,
    n_repeats,
    data_dir,
    save_dir,
    kernel,
    inducing,
    likelihood,
    n_iter,
    log_frequency,
    n_inducing_points,
    gp_model,
    settings=None,
    **kwargs,
):

    # Init dataclass
    dataset = SingleResolutionDataset(**settings)
    dataset.split_data()
    train_data = dataset.train_data

    model = NarcolepsyModel(**settings)

    # Run cross-validation
    fold_count = -1
    cv_eval_preds = dict()
    cv_eval_preds["mean"] = np.zeros((n_repeats, len(y), 1))
    cv_eval_preds["var"] = np.zeros((n_repeats, len(y), 1))
    cv_eval_preds["pred_class"] = np.zeros((n_repeats, len(y), 1))
    cv_eval_preds["true_class"] = np.zeros((n_repeats, len(y)))
    cv_eval_preds["train_idx"] = [[[] for _ in range(n_kfold)] for _ in range(n_repeats)]
    cv_eval_preds["test_idx"] = [[[] for _ in range(n_kfold)] for _ in range(n_repeats)]
    cv_train_preds = dict()
    cv_train_preds["mean"] = np.zeros((n_repeats, n_kfold, len(y), 1))
    cv_train_preds["var"] = np.zeros((n_repeats, n_kfold, len(y), 1))
    elbo = [[[] for j in range(n_kfold)] for i in range(n_repeats)]
    models = [[[] for j in range(n_kfold)] for i in range(n_repeats)]
    scalers = [[[] for j in range(n_kfold)] for i in range(n_repeats)]
    pcas = (
        [[{"obj": None, "n_components": None, "components": None} for j in range(n_kfold)] for i in range(n_repeats)]
        if options["pca"]
        else None
    )
    cv_acc = []
    cv_auc = []
    mean = []
    var = []
    for (current_repeat, current_fold), (train, test) in zip(
        itertools.product(range(n_repeats), range(n_kfold)), skf.split(X, y)
    ):
        fold_count += 1
        logger.info(f"Current repeat: {current_repeat + 1}/{n_repeats}")
        logger.info(f"Current fold: {current_fold + 1}/{n_kfold}")

        # Index into data
        X_train = train_data.X[train, :]
        y_train = train_data.y[train]
        X_test = train_data.X[test, :]
        y_test = train_data.y[test]
        cv_eval_preds["train_idx"][current_repeat][current_fold] = train
        cv_eval_preds["test_idx"][current_repeat][current_fold] = test

        # Run scaling
        scalers[current_repeat][current_fold] = dataset.scaler
        X_trainer = dataset.scale_features(X_train)
        X_test = dataset.scale_features(X_test)
        dataset.clear_scaler()
        
        # RobustScaler(quantile_range=(15.0, 85.0)).fit(X_train)
        # X_train = scalers[current_repeat][current_fold].transform(X_train)
        # X_test = scalers[current_repeat][current_fold].transform(X_test)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--kernel", type=str, default="rbf")
    parser.add_argument("-d", "--data_dir", default="data/narco_features", type=str)
    parser.add_argument("-f", "--feature_set", default="all", type=str)
    parser.add_argument("-i", "--inducing", type=str, default="data", choices=["data", "kmeans"])
    parser.add_argument("-k", "--n_kfold", default=5, type=int)
    parser.add_argument("-l", "--likelihood", type=str, default="gaussian", choices=["bernoulli", "gaussian"])
    parser.add_argument("-m", "--model", required=True, type=str)
    parser.add_argument("-n", "--n_repeats", default=1, type=int)
    parser.add_argument("-p", "--n_inducing_points", default=350, type=int)
    parser.add_argument("-r", "--resolution", type=float)
    parser.add_argument("-s", "--save_dir", type=str)
    parser.add_argument("--data-master", type=str, default="data_master.csv")
    parser.add_argument("--gp_model", type=str, default="svgp")
    parser.add_argument("--log_frequency", type=int, default=10)
    parser.add_argument("--n_iter", type=int, default=10000)
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--pca_components", default=100)
    parser.add_argument("--pca_explained", default=90)
    parser.add_argument("--feature_selection", action="store_true")
    parser.add_argument("--smote", dest="smote", action="store_true")
    parser.add_argument("--no-smote", dest="smote", action="store_false")
    parser.add_argument("--seed", default=42, help='Seed for random number generators.')
    parser.set_defaults(smote=True)
    args = parser.parse_args()

    # Set seed
    os.environ["PL_GLOBAL_SEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

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

    train_model(**vars(args), options=vars(args))
    print("fin")
