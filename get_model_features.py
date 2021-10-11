import os
from argparse import ArgumentParser

import gpflow as gpf
import numpy as np
import tensorflow as tf

from feature_descriptions import get_feature_descriptions


def get_model_features(model_path):

    model_folds = sorted(os.listdir(os.path.join(model_path, "models")))
    l_values = []
    l_sorted_idx = []
    custom_config = gpf.settings.get_settings()
    custom_config.verbosity.tf_compile_verb = True
    with gpf.session_manager.get_session().as_default():
        for current_model_fold in model_folds:
            m = gpf.saver.Saver().load(
                os.path.join(model_path, "models", current_model_fold)
            )
            l_values.append(m.kern.lengthscales.value)
            l_sorted_idx.append(np.argsort(m.kern.lengthscales.value))
    del m
    l_values = np.sqrt(np.asarray(l_values))  # l
    l_sorted_idx = np.asarray(l_sorted_idx)
    l_values_scaled = l_values.mean(axis=0) / l_values.mean(axis=0).min()
    get_feature_descriptions(
        np.argsort(l_values.mean(axis=0) / l_values.mean(axis=0).min())
    )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True)
    args = parser.parse_args()

    model_path = os.path.join("experiments", args.model)
    get_model_features(model_path)
