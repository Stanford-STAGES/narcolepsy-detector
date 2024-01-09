import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import gpflow as gpf
import numpy as np
from imblearn.over_sampling import SMOTE
from scipy.cluster.vq import kmeans
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
import tensorflow as tf

from utils.logger import get_logger


logger = get_logger()


@dataclass
class NarcolepsyModel:
    """
    NarcolepsyModel:
    """

    gp_model: Literal["vgp", "svgp"] = "svgp"
    inducing: Literal["kmeans", "data"] = "kmeans"
    kernel: str = "rbf"
    likelihood: str = "bernoulli"
    n_inducing_points: int = 350
    n_iter: int = 10_000
    log_frequency: int = 10
    pca: bool = False
    pca_components: int = 100
    pca_explained: int = 90
    smote: bool = True

    def __post_init__(self):

        if self.smote:
            self.smote_object = SMOTE(k_neighbors=10, n_jobs=-1)

        self.custom_config = gpf.config.Config(jitter=1e-6)

        self.kernels = dict(
            rbf=gpf.kernels.SquaredExponential,
            rq=gpf.kernels.RationalQuadratic,
        )
        self.likelihoods = dict(
            bernoulli=gpf.likelihoods.Bernoulli(),
            gaussian=gpf.likelihoods.Gaussian(),
        )
        self.thresholds = dict(bernoulli=0.5, gaussian=0.0)

    def fit(self, X_train, y_train):

        if self.pca:
            self._fit_pca(X_train)
            X_train = self._run_pca(X_train)

        if self.smote:
            X_train, y_train = self._run_smote(X_train, y_train)

        inducing_points = self._get_inducing_points(X_train, y_train)

        data = (X_train, y_train)
        self.data_shape = X_train.shape
        N, M = self.data_shape

        logger.info(f"Training GP model with input data.shape=({N}, {M})")
        restart = True
        custom_config = self.custom_config
        while restart:
            with gpf.config.as_context(custom_config):

                # Create model instance
                logger.info(f"\tUsing {self.gp_model} for variational inference")
                if self.gp_model == "svgp":
                    m = gpf.models.SVGP(
                        kernel=self.kernels[self.kernel](lengthscales=np.ones(M)),
                        likelihood=self.likelihoods[self.likelihood],
                        inducing_variable=inducing_points,
                        num_data=N,
                    )
                    objective = m.training_loss_closure(data)
                elif self.gp_model == "vgp":
                    m = gpf.models.VGP(
                        data=data,
                        kernel=self.kernels[self.kernel](lengthscales=np.ones(M)),
                        likelihood=self.likelihoods[self.likelihood],
                    )
                    objective = m.training_loss

                # Create optimizer
                optimizer = gpf.optimizers.Scipy()

                @tf.function(reduce_retracing=True)
                def calculate_elbo():
                    return -objective()

                logger.info(f"\tBeginning optimization of GP model with {self.kernel} kernel and {self.likelihood} likelihood")
                elbo_init = calculate_elbo().numpy()
                logger.info(f"\tELBO before optimization: {elbo_init}")
                start = time.time()

                try:
                    optimizer.minimize(objective, m.trainable_variables, options=dict(maxiter=self.n_iter))
                    if calculate_elbo().numpy() > elbo_init:
                        end = time.time()
                        restart = False
                    else:
                        new_jitter, old_jitter = self._update_jitter()
                        custom_config = gpf.config.Config(jitter=new_jitter)
                        logger.info(
                            f"\tCaptured an error, most likely in the Cholesky decomp. Raising jitter levels from {old_jitter} to {new_jitter} and retrying"
                        )
                except:
                    new_jitter, old_jitter = self._update_jitter()
                    custom_config = gpf.config.Config(jitter=new_jitter)
                    logger.info(
                        f"\tCaptured an error, most likely in the Cholesky decomp. Raising jitter levels from {old_jitter} to {new_jitter} and retrying"
                    )
        logger.info(f"\tElapsed time: {end - start}")
        logger.info(f"\tELBO after optimization: {calculate_elbo().numpy()}")

        self._model = m

    @property
    def model(self):
        return self._model

    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None):

        mean_pred, var_pred = self.model.predict_y(X)

        return mean_pred.numpy(), var_pred.numpy()

    def save_model(self, savepath: Path) -> None:
        logger.info(f"Saving model at {savepath}")
        self.model.predict_y_compiled = tf.function(
            self.model.predict_y, input_signature=[tf.TensorSpec(shape=[None, self.data_shape[1]], dtype=tf.float64)]
        )
        tf.saved_model.save(self.model, savepath)
        settings = {
            key: getattr(self, key)
            for key in (
                "gp_model",
                "inducing",
                "kernel",
                "likelihood",
                "n_inducing_points",
                "n_iter",
                "log_frequency",
                "pca",
                "pca_components",
                "pca_explained",
                "smote",
                "data_shape",
            )
        }
        with open(savepath / "settings.json", "w") as fp:
            json.dump(settings, fp)

    @classmethod
    def load_saved_model(cls, model_path):
        with open(model_path / "settings.json", "r") as fp:
            settings = json.load(fp)
        data_shape = settings.pop("data_shape")
        model = cls(**settings)
        model._model = tf.saved_model.load(model_path)
        model.data_shape = data_shape

        return model

    def evaluate_performance(self, mu_pred, y_true):
        y_pred = np.ones(mu_pred.shape)
        y_pred[mu_pred < self.thresholds[self.likelihood]] = 0
        acc = np.mean(np.squeeze(y_pred) == np.squeeze(y_true.squeeze()))
        auc = roc_auc_score(y_true.squeeze(), mu_pred)
        return acc, auc, y_pred

    def _fit_pca(self, X):

        logger.info("Running PCA:")
        n_components = self.pca_components
        while True:
            pca = PCA(n_components=n_components, whiten=True).fit(X)
            explained_var = np.cumsum(pca.explained_variance_ratio_ * 100)
            if explained_var[-1] >= self.pca_explained:
                if explained_var[-2] < self.pca_explained:
                    logger.info(f"\tPCA successful: {n_components} components accounting for {explained_var[-1]:.1f}% variance")
                    break
                else:
                    logger.info("\tPCA running with excess components, adjusting n_components")
                    n_components = np.argwhere(explained_var >= self.pca_explained)[0][0] + 1
            else:
                logger.info("\tNot enough variance explained, adjusting n_components")
                n_components *= 2

        self.pca_object = pca

    def _run_pca(self, X):

        if hasattr(self, "pca_object"):
            logger.info("Transforming with PCA")
            X = self.pca_object.transform(X)

        return X

    def _run_smote(self, X, y):

        logger.info("Running SMOTE:")
        X, y = self.smote_object.fit_resample(X, y)
        logger.info(f"\tShape of X, resampled: {X.shape}")

        return X, y

    def _get_inducing_points(self, X, y):

        if self.inducing == "data":
            logger.info("\tUsing existing data points as inducing points")
            inducing_points = np.arange(len(y))
            np.random.shuffle(inducing_points)
            inducing_points = inducing_points[: self.n_inducing_points]
            inducing_points = X[inducing_points, :]
        elif self.inducing == "kmeans":
            logger.info("\tUsing kmeans to calculate inducing points")
            inducing_points = kmeans(X, self.n_inducing_points)[0]

        return inducing_points

    @staticmethod
    def _update_jitter():

        current_jitter = gpf.config.config().jitter
        current_jitter_exponent = np.floor(np.log10(np.abs(current_jitter)))
        new_jitter_exponent = current_jitter_exponent + 1
        new_jitter = 10**new_jitter_exponent

        return new_jitter, current_jitter

    # def __init__(
    #     self,
    #     resolution: float,
    #     save_dir: str,
    #     feature_selection: bool = True,
    #     feature_set: str = "all",
    #     gp_model: str = "svgp",
    #     inducing: str = "kmean",
    #     kernel: str = "rbf",
    #     likelihood: str = "bernoulli",
    #     model: str = "usleep",
    #     n_inducing_points: int = 350,
    #     n_iter: int = 10_000,
    #     n_kfold: int = 5,
    #     n_repeats: int = 5,
    #     log_frequency: int = 10,
    #     pca: bool = False,
    #     pca_components: int = 100,
    #     pca_explained: int = 90,
    #     smote: bool = True,
    # ) -> None:
    #     self.resolution = resolution
    #     self.save_dir = save_dir
    #     self.feature_selection = feature_selection
    #     self.feature_set = feature_set
    #     self.gp_model = gp_model
    #     self.inducing = inducing
    #     self.kernel = kernel
    #     self.likelihood = likelihood
    #     self.model = model
    #     self.n_inducing_points = n_inducing_points
    #     self.n_iter = n_iter
    #     self.n_kfold = n_kfold
    #     self.n_repeats = n_repeats
    #     self.log_frequency = log_frequency
    #     self.pca = pca
    #     self.pca_components = pca_components
    #     self.pca_explained = pca_explained
    #     self.smote = smote

    #     self.logger = get_logger()

    #     self._initialize_model_list()
    #     self._initialize_scalers_list()

    # def _initialize_model_list(self) -> None:
    #     models = [[[] for j in range(self.n_kfold)] for i in range(self.n_repeats)]

    # def _initialize_scalers_list(self) -> None:
    #     scalers = [[[] for j in range(self.n_kfold)] for i in range(self.n_repeats)]

    # def _initialize_crossval_proc(self, N: int) -> None:

    #     self.skf = model_selection.RepeatedStratifiedKFold(n_splits=self.n_kfold, n_repeats=self.n_repeats)
    #     self.fold_count = -1
    #     self.cv_eval_preds = dict()
    #     self.cv_eval_preds["mean"] = np.zeros((self.n_repeats, N, 1))
    #     self.cv_eval_preds["var"] = np.zeros((self.n_repeats, N, 1))
    #     self.cv_eval_preds["pred_class"] = np.zeros((self.n_repeats, N, 1))
    #     self.cv_eval_preds["true_class"] = np.zeros((self.n_repeats, N))
    #     self.cv_eval_preds["train_idx"] = [[[] for _ in range(self.n_kfold)] for _ in range(self.n_repeats)]
    #     self.cv_eval_preds["test_idx"] = [[[] for _ in range(self.n_kfold)] for _ in range(self.n_repeats)]
    #     self.cv_train_preds = dict()
    #     self.cv_train_preds["mean"] = np.zeros((self.n_repeats, self.n_kfold, N, 1))
    #     self.cv_train_preds["var"] = np.zeros((self.n_repeats, self.n_kfold, N, 1))
    #     self.cv_acc = []
    #     self.cv_auc = []
    #     self.mean = []
    #     self.var = []

    # def fit(self, X: np.ndarray, y: np.ndarray):

    #     self._initialize_crossval_proc(len(y))

    #     for (current_repeat, current_fold), (train, test) in zip(
    #         itertools.product(range(self.n_repeats), range(self.n_kfold)), self.skf.split(X, y)
    #     ):
    #         self.fold_count += 1
    #         self.logger.info(f"Current repeat: {current_repeat + 1}/{self.n_repeats}")
    #         self.logger.info(f"Current fold: {current_fold + 1}/{self.n_kfold}")

    #         # Index into data
    #         X_train = X[train, :]
    #         y_train = y[train]
    #         X_test = X[test, :]
    #         y_test = y[test]
    #         self.cv_eval_preds["train_idx"][current_repeat][current_fold] = train
    #         self.cv_eval_preds["test_idx"][current_repeat][current_fold] = test

    #         # Run scaling
    #         self.scalers[current_repeat][current_fold] = RobustScaler(quantile_range=(15.0, 85.0)).fit(X_train)
    #         X_train = self.scalers[current_repeat][current_fold].transform(X_train)
    #         X_test = self.scalers[current_repeat][current_fold].transform(X_test)

    #         if options["feature_selection"]:
    #             logger.info("Running feature selection procedure for current fold")

    #             def _feature_selection(features, labels, n_folds):
    #                 features_squared = np.square(features)
    #                 _X = np.concatenate([features, features_squared], axis=1)
    #                 _y = labels
    #                 skf = skl.model_selection.StratifiedKFold(n_splits=n_folds)
    #                 # estimator = skl.svm.SVC(kernel="linear")
    #                 # estimator = skl.svm.SVC(kernel="rbf")
    #                 estimator = skl.ensemble.RandomForestClassifier()
    #                 rfecv = RFECV(estimator=estimator, step=1, cv=skf, scoring="accuracy", n_jobs=-1, verbose=0)
    #                 logger.info("Starting RFE procedure")
    #                 start = time.time()
    #                 rfecv.fit(_X, _y)
    #                 end = time.time()
    #                 relevant = rfecv.get_support()
    #                 score = max(rfecv.grid_scores_)
    #                 logger.info(f"\tElapsed time: {end-start}")
    #                 logger.info(f"\tOptimal number of features: {rfecv.n_features_}")
    #                 logger.info(
    #                     f"\tOptimal number of unique features: {sum(relevant[:len(relevant)//2] + relevant[len(relevant)//2:])}"
    #                 )

    #                 # Save selected features
    #                 save_file_path = os.path.join(
    #                     save_dir, "feature-selection", f"selected-features_n{current_repeat+1:02}_k{current_fold+1:02}.pkl"
    #                 )
    #                 os.makedirs(os.path.join(save_dir, "feature-selection"), exist_ok=True)
    #                 logger.info(f"Saving objects at {save_file_path}")
    #                 selected_features = {"features": relevant, "score": score, "model": rfecv}
    #                 with open(save_file_path, "wb") as fp:
    #                     pickle.dump(selected_features, fp)

    #                 selected_features = np.array(relevant).astype(int)
    #                 selected_features = (
    #                     selected_features[: len(selected_features) // 2] + selected_features[len(selected_features) // 2 :]
    #                 ) > 0

    #                 return selected_features

    #             selected_features = _feature_selection(X_train, y_train, 10)
    #             X_train = X_train[:, selected_features]
    #             X_test = X_test[:, selected_features]
    #             logger.info(f"Using {X_train.shape[1]} features")

    #         # Do PCA. We select the number of components that explain 90% of the variance
    #         if options["pca"]:
    #             logger.info("Running PCA")
    #             n_components = options["pca_components"]
    #             while True:
    #                 logger.info(f"\tFitting to train data using {n_components} components")
    #                 pca = PCA(n_components=n_components, whiten=True).fit(X_train)
    #                 explained_var = np.cumsum(pca.explained_variance_ratio_ * 100)
    #                 if explained_var[-1] >= options["pca_explained"]:
    #                     if explained_var[-2] < options["pca_explained"]:
    #                         logger.info(
    #                             f"\tPCA successful: {n_components} components accounting for {explained_var[-1]:.1f}% variance"
    #                         )
    #                         break
    #                     else:
    #                         logger.info("\tPCA running with excess components, adjusting n_components")
    #                         n_components = np.argwhere(explained_var >= options["pca_explained"])[0][0] + 1
    #                 else:
    #                     logger.info("\tNot enough variance explained, adjusting n_components")
    #                     n_components *= 2

    #             logger.info("\tTransforming train and test partitions with PCA")
    #             X_train = pca.transform(X_train)
    #             X_test = pca.transform(X_test)
    #             N, M = X_train.shape
    #             pcas[current_repeat][current_fold]["obj"] = pca
    #             pcas[current_repeat][current_fold]["components"] = pca.components_
    #             pcas[current_repeat][current_fold]["n_components"] = n_components

    #         # Run SMOTE to oversample narcolepsy class
    #         # print('{} | Running SMOTE'.format(datetime.now()))
    #         if options["smote"]:
    #             logger.info("Running SMOTE")
    #             # X_res, y_res = SMOTE(k_neighbors=10, m_neighbors=20, n_jobs=-1).fit_sample(X_train, y_train)
    #             X_res, y_res = SMOTE(k_neighbors=10, n_jobs=-1).fit_resample(X_train, y_train)
    #             logger.info(f"\tShape of X: {X_res.shape}")
    #         else:
    #             X_res, y_res = X_train, y_train

    #         # Set induction points
    #         if inducing == "data":
    #             logger.info("Using data for inducing points")
    #             inductionP = np.arange(len(y_train))
    #             np.random.shuffle(inductionP)
    #             inductionP = inductionP[:num_induction_points]
    #             inductionP = X_res[inductionP, :]
    #         elif inducing == "kmeans":
    #             from scipy.cluster.vq import kmeans

    #             logger.info("Using kmeans for inducing points")
    #             inductionP = kmeans(X_res, num_induction_points)[0]

    #         # Here we run the GP optimization
    #         # kernel = kernels.RBF(np.ones(len(selected_features)))
    #         # m_fix = GaussianProcessClassifier(kernel=kernel, n_jobs=-1, max_iter_predict=500, optimizer=None)
    #         # m_fix.fit(X_res, y_res)
    #         # m = GaussianProcessClassifier(kernel=kernel, n_jobs=-1, max_iter_predict=500)

    #         # New GPflow
    #         data = (X_res, y_res[:, np.newaxis])
    #         # data_minibatch = tf.data.Dataset.from_tensor_slices(data).repeat().shuffle(10000).batch(100).prefetch(tf.data.AUTOTUNE)
    #         # data_minibatch_it = iter(data_minibatch)

    #         custom_config = gpf.config.Config(jitter=1e-6)

    #         # custom_config = gpf.settings.get_settings()
    #         # custom_config.verbosity.tf_compile_verb = True
    #         # current_jitter = custom_config.numerics.jitter_level
    #         # current_jitter_magnitude = -6
    #         # custom_config.numerics.jitter_level = 10 ** current_jitter_magnitude
    #         restart = True
    #         while restart:
    #             # try:
    #             # with gpf.settings.temp_settings(custom_config), gpf.session_manager.get_session().as_default():
    #             with gpf.config.as_context(custom_config):

    #                 # Create model
    #                 logger.info(f"Using {gp_model} for variational inference")
    #                 if gp_model == "svgp":
    #                     m = gpf.models.SVGP(
    #                         kernel=kernels[kernel](lengthscales=np.ones(X_res.shape[1])),
    #                         likelihood=likelihoods[likelihood],
    #                         inducing_variable=inductionP,
    #                         num_data=X_res.shape[0],
    #                     )
    #                     # objective = m.training_loss_closure(data_minibatch_it)
    #                     objective = m.training_loss_closure(data)
    #                 elif gp_model == "vgp":
    #                     m = gpf.models.VGP(
    #                         data,
    #                         kernel=kernels[kernel](lengthscales=np.ones(X_res.shape[1])),
    #                         likelihood=likelihoods[likelihood],
    #                     )
    #                     objective = m.training_loss

    #                 # Create optimizer
    #                 optimizer = gpf.optimizers.Scipy()
    #                 # gpf.set_trainable(m.q_mu, False)
    #                 # gpf.set_trainable(m.q_sqrt, False)
    #                 # optimizer = tf.optimizers.Adam(0.001)
    #                 # adam_opt = tf.optimizers.Adam(0.001)
    #                 # nat_opt = gpf.optimizers.NaturalGradient(gamma=0.1)
    #                 # variational_params = [(m.q_mu, m.q_sqrt)]

    #                 @tf.function
    #                 def calculate_elbo():
    #                     return -objective()

    #                 # cv_logdir = os.path.join(tf_logdir, f"cv-{current_repeat:02}-{current_fold:02}")
    #                 # model_task = ModelToTensorBoard(cv_logdir, m)
    #                 # lml_task = ScalarToTensorBoard(cv_logdir, calculate_elbo, "elbo")
    #                 # tasks = MonitorTaskGroup([model_task, lml_task], period=log_frequency)
    #                 # monitor = Monitor(tasks)

    #                 # @tf.function
    #                 # def optimization_step(i):
    #                 # optimizer.minimize(objective, m.trainable_variables)
    #                 # monitor(i)
    #                 # adam_opt.minimize(objective, var_list=m.trainable_variables)
    #                 # nat_opt.minimize(objective, var_list=variational_params)

    #                 # m = gpf.models.VGP(
    #                 #     data=(X_res, y_res),
    #                 #     kernel=kernels[kernel](lengthscales=np.ones(len(selected_features))),
    #                 #     likelihood=gpf.likelihoods.Bernoulli(),
    #                 # )
    #                 # optimizer = gpf.optimizers.Scipy()

    #                 # variational_params = [(m.q_mu, m.q_sqrt)]
    #                 # Old GPflow
    #                 # m = gpf.models.SVGP(
    #                 #     X_res,
    #                 #     y_res[:, np.newaxis],
    #                 #     kern=kernels[kernel](len(selected_features), ARD=True),
    #                 #     likelihood=likelihoods[likehood],
    #                 #     Z=inductionP,
    #                 # )

    #                 # if kernel == "rbf":
    #                 #     m = gpf.models.SVGP(
    #                 #         X_res,
    #                 #         y_res[:, np.newaxis],
    #                 #         kern=gpf.kernels.RBF(len(selected_features), ARD=True),
    #                 #         likelihood=gpf.likelihoods.Gaussian(),
    #                 #         Z=inductionP,
    #                 #     )
    #                 # elif kernel == "rq":
    #                 #     m = gpf.models.SVGP(
    #                 #         X_res,
    #                 #         y_res[:, np.newaxis],
    #                 #         kern=gpf.kernels.RationalQuadratic(len(selected_features), ARD=True),
    #                 #         likelihood=gpf.likelihoods.Gaussian(),
    #                 #         Z=inductionP,
    #                 #     )
    #                 # m = gpf.models.GPRFITC(X_res, y_res.reshape((-1, 1)), kern=gpf.kernels.RBF(len(selected_features), ARD=True), Z=inductionP)
    #                 # print('{} | Starting optimization of GP model | Negative log likelihood before optimization: {}'.format(datetime.now(), m.compute_log_likelihood()))
    #                 logger.info(f"Starting optimization of GP model with {kernel} kernel and {likelihood} likelihood")
    #                 # logger.info(f"\tELBO before optimization: {m.elbo(data).numpy()}")
    #                 elbo_init = calculate_elbo().numpy()
    #                 logger.info(f"\tELBO before optimization: {elbo_init}")
    #                 # logger.info(f"\tNegative log likelihood before optimization: {m.compute_log_likelihood()}")
    #                 # print('{} | Starting optimization of GP model | Negative log likelihood before optimization: {}'.format(datetime.now(), m_fix.log_marginal_likelihood()))
    #                 start = time.time()
    #                 # m.fit(X_res, y_res)
    #                 # optimizer.minimize(training_loss, m.trainable_variables, options=dict(maxiter=100000))
    #                 # bar = tqdm(range(1000))
    #                 # for i in tqdm(tf.range(n_iter)):

    #                 # with trange(n_iter) as t:
    #                 #     for step in t:
    #                 # optimizer.minimize(objective, m.trainable_variables)
    #                 # optimization_step(current_fold)
    #                 try:
    #                     optimizer.minimize(objective, m.trainable_variables, options=dict(maxiter=n_iter))
    #                     # if step % log_frequency == 0:
    #                     #     elbo[current_repeat][current_fold].append(calculate_elbo(fold_count).numpy())
    #                     #     t.set_postfix(elbo=elbo[current_repeat][current_fold][-1])
    #                     # fold_run.log({"step": step, "ELBO": elbo[current_repeat][current_fold][-1]})
    #                     # adam_opt.minimize(objective, var_list=m.trainable_variables)
    #                     # nat_opt.minimize(objective, var_list=variational_params)
    #                     # t.set_postfix(nll=m.elbo(data).numpy())
    #                     # gpf.train.ScipyOptimizer().minimize(m, maxiter=100000)
    #                     if calculate_elbo().numpy() > elbo_init:
    #                         end = time.time()
    #                         restart = False
    #                     else:
    #                         current_jitter = gpf.config.config().jitter
    #                         current_jitter_exponent = np.floor(np.log10(np.abs(current_jitter)))
    #                         new_jitter_exponent = current_jitter_exponent + 1
    #                         new_jitter = 10 ** new_jitter_exponent
    #                         custom_config = gpf.config.Config(jitter=new_jitter)
    #                         logger.info(
    #                             f"\tCaptured an error, most likely in the Cholesky decomp. Raising jitter levels from {current_jitter} to {new_jitter} and retrying"
    #                         )
    #                 except:
    #                     current_jitter = gpf.config.config().jitter
    #                     current_jitter_exponent = np.floor(np.log10(np.abs(current_jitter)))
    #                     new_jitter_exponent = current_jitter_exponent + 1
    #                     new_jitter = 10 ** new_jitter_exponent
    #                     custom_config = gpf.config.Config(jitter=new_jitter)
    #                     logger.info(
    #                         f"\tCaptured an error, most likely in the Cholesky decomp. Raising jitter levels from {current_jitter} to {new_jitter} and retrying"
    #                     )
    #             # except:
    #             #     current_jitter = 10 ** current_jitter_magnitude
    #             #     new_jitter_magnitude = current_jitter_magnitude + 1
    #             #     new_jitter = 10 ** new_jitter_magnitude
    #             #     logger.info(
    #             #         f"\tCaptured an error, most likely in the Cholesky decomp. Raising jitter levels from {current_jitter} to {new_jitter} and retrying"
    #             #     )
    #             #     current_jitter_magnitude = new_jitter_magnitude
    #             #     current_jitter = new_jitter
    #             #     custom_config.numerics.jitter_level = current_jitter
    #             #     # with gpf.settings.temp_settings(custom_config), gpf.session_manager.get_session().as_default():
    #             #     #     if kernel == 'rbf':
    #             #     #         m = gpf.models.SVGP(X_res, y_res[:, np.newaxis], kern=gpf.kernels.RBF(len(selected_features), ARD=True), likelihood=gpf.likelihoods.Gaussian(), Z=inductionP)
    #             #     #     elif kernel == 'rq':
    #             #     #         m = gpf.models.SVGP(X_res, y_res[:, np.newaxis], kern=gpf.kernels.RationalQuadratic(len(selected_features), ARD=True), likelihood=gpf.likelihoods.Gaussian(), Z=inductionP)
    #             #     #     # m = gpf.models.GPRFITC(X_res, y_res.reshape((-1, 1)), kern=gpf.kernels.RBF(len(selected_features), ARD=True), Z=inductionP)
    #             #     #     # print('{} | Starting optimization of GP model | Negative log likelihood before optimization: {}'.format(datetime.now(), m.compute_log_likelihood()))
    #             #     #     logger.info(f'Starting optimization of GP model with {kernel} kernel')
    #             #     #     logger.info(f'\tNegative log likelihood before optimization: {m.compute_log_likelihood()}')
    #             #     #     start = time.time()
    #             #     #     gpf.train.ScipyOptimizer().minimize(m, maxiter=100000)
    #             #     #     end = time.time()
    #         logger.info(f"\tElapsed time: {end - start}")
    #         # logger.info(f"\tNegative log likelihood after optimization: {m.compute_log_likelihood()}")
    #         # logger.info(f"\tTraining loss after optimization: {m.training_loss(data=(X_res, y_res[:, np.newaxis]))}")
    #         # logger.info(f"\tELBO after optimization: {m.elbo(data).numpy()}")
    #         logger.info(f"\tELBO after optimization: {calculate_elbo().numpy()}")
    #         # print('{} | Elapsed time: {} | Negative log likelihood after optimization: {}'.format(datetime.now(), end - start, m.compute_log_likelihood()))
    #         # print('{} | Elapsed time: {} | Negative log likelihood after optimization: {}'.format(datetime.now(), end - start, m.log_marginal_likelihood()))

    #         # Get predictions
    #         mean_pred, var_pred = m.predict_y(X_test)

    #         # Get the prediction accuracy at 50% cutoff
    #         y_pred = np.ones(mean_pred.shape)
    #         y_pred[mean_pred < THRESHOLD[likelihood]] = -1
    #         acc = np.mean(np.squeeze(y_pred) == np.squeeze(y_test))
    #         cv_acc.append(acc)
    #         logger.info(f"\tFold accuracy: {acc}")

    #         # Save the model and save predictions
    #         cv_auc.append(roc_auc_score(y_test, mean_pred))
    #         models[current_repeat][current_fold] = m
    #         cv_eval_preds["mean"][current_repeat, test] = mean_pred
    #         cv_eval_preds["var"][current_repeat, test] = var_pred
    #         cv_eval_preds["pred_class"][current_repeat, test] = y_pred
    #         cv_eval_preds["true_class"][current_repeat, test] = y_test

    #         model_savepath = os.path.join(model_path, f"model_n{current_repeat + 1:02}_k{current_fold + 1:02}") + ".gpm"
    #         logger.info(f"\tSaving model {model_savepath}")
    #         m.predict_y_compiled = tf.function(
    #             m.predict_y, input_signature=[tf.TensorSpec(shape=[None, X_train.shape[1]], dtype=tf.float64)]
    #         )
    #         tf.saved_model.save(m, model_savepath)

    #         # ROC CURVES ON TRAINING DATA
    #         plot_roc(
    #             y_test,
    #             mean_pred,
    #             THRESHOLD[likelihood],
    #             f"CV eval data {current_repeat+1:02}/{current_fold+1:02}",
    #             savepath=os.path.join(figure_path, f"roc_n{current_repeat+1:02}_k{current_fold+1:02}.png"),
    #         )

    #         # Get predictions on entire training set
    #         if options["pca"]:
    #             mean_train, var_train = m.predict_y(
    #                 pcas[current_repeat][current_fold]["obj"].transform(
    #                     scalers[current_repeat][current_fold].transform(X)[:, selected_features]
    #                 )
    #             )
    #         else:
    #             mean_train, var_train = m.predict_y(scalers[current_repeat][current_fold].transform(X)[:, selected_features])
    #         cv_train_preds["mean"][current_repeat, current_fold, :] = mean_train
    #         cv_train_preds["var"][current_repeat, current_fold, :] = var_train
    #         mean.append(mean_train)
    #         var.append(var_train)

    #         logger.info(f"\tCross-validation accuracy: {np.mean(cv_acc)} +/- {np.std(cv_acc)}\n")
    #         logger.info(f"\tCross-validation AUC: {np.mean(cv_auc)} +/- {np.std(cv_auc)}\n")

    #         # del m, training_loss, optimizer
