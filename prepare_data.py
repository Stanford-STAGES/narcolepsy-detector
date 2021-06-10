import csv
import os
import pandas as pd
import pickle
from argparse import ArgumentParser
from datetime import datetime
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from tqdm import tqdm

# import extract_features
from inf_extract_features import ExtractFeatures
from utils import ParallelExecutor
from utils import match_data
from utils import rolling_window


def load_csv():
    # N = '/home/jens/Documents/stanford/overview_file_cohorts.csv'
    N = "./data_master.csv"
    trainL = []
    label = []
    ID = []
    ID_ling = []
    df = pd.read_csv(N)
    with open(N) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:

            # trainL += [int(row['Used for narco training'])]
            trainL += [int(row["Narcolepsy training data"])]
            ID += [row["ID"]]
            ID_ling += [row["ID-ling"]]
            label += [int(row["Label"])]

    return ID, ID_ling, label, trainL, df


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="./data/massc")
    parser.add_argument("-r", "--resolution", type=int, default=15)
    args = parser.parse_args()

    ### SHERLOCK PATHS
    # sherlock_scratch_path = '/scratch/users/alexno/narco_ml'
    # resPath = os.path.join(sherlock_scratch_path, 'sc_predictions')
    # scalingPath = os.path.join(sherlock_scratch_path, 'narco_scaling')
    # featurePath = os.path.join(sherlock_scratch_path, 'narco_features')
    # sleep_predictions_path = './data/massc'
    sleep_predictions_path = args.data_dir
    narco_scaling_path = "./data/narco_scaling"
    narco_feature_path = "./data/narco_features"

    for p in [sleep_predictions_path, narco_scaling_path, narco_feature_path]:
        if not os.path.exists(p):
            print("{} | Creating directory: {}".format(datetime.now(), p))
            os.makedirs(p)
    config = type(
        "obj",
        (object,),
        {
            "narco_feature_selection_path": None,
            "narco_scaling_path": narco_scaling_path,
            "resolution": args.resolution,
        },
    )  # This is just a dummy object, since I don't want to use the config file
    ###

    extract_features = ExtractFeatures(config)

    foldersPath = os.listdir(sleep_predictions_path)
    model_str = os.path.basename(sleep_predictions_path)
    # [foldersPath.remove(model) for model in foldersPath if model[:3] != 'ac_']
    if "archive" in foldersPath:
        foldersPath.remove("archive")
    foldersPath.sort()

    count = -1
    if True:
        # for f in foldersPath:
        # count += 1
        # filesPath = os.listdir(os.path.join(sleep_predictions_path, f, ))
        # filesPath = glob(os.path.join(sleep_predictions_path, f, "**", "pred*.pkl"), recursive=True)
        filesPath = glob(os.path.join(sleep_predictions_path, "**", "pred*.pkl"), recursive=True)
        DF = match_data(filesPath)
        # DF = DF[DF["Narcolepsy training data"] == 1]

        # DEBUG
        # DF = DF[:10]

        features = np.zeros([24 + 31 * 15, len(DF)])
        labels = np.zeros(len(DF))
        ID = [[]] * len(DF)
        cohort = [[]] * len(DF)
        for i in tqdm(range(len(DF))):
            # print('{} | {} of {} files'.format(datetime.now(), str(i), str(len(DF))))
            # with open(DF.iloc[i, 0], "rb") as pkl:
            with open(DF.iloc[i].Filepath, "rb") as pkl:
                contents = pickle.load(pkl)
            # contents = sio.loadmat(os.path.join(sleep_predictions_path, f, DF.iloc[i, 0]), squeeze_me=True, struct_as_record=False)
            labels[i] = DF["label"].values[i]
            ID[i] = DF.iloc[i, 1]
            cohort[i] = DF.iloc[i]["Cohort"]
            # contents = contents['predictions']
            if args.resolution == 1:
                resolution_key = "logits"
            elif args.resolution == 30:
                resolution_key = "predicted"
            else:
                # elif args.resolution in [3, 5, 10, 15]:
                resolution_key = f"yhat_{args.resolution}s"
            try:
                hypnodensity = contents[resolution_key]
            except:
                hypnodensity = contents["logits"]
                hypnodensity = rolling_window(hypnodensity, args.resolution, args.resolution).mean(axis=1)
                # N, K = hypnodensity.shape
                # hypnodensity = hypnodensity.reshape(-1, args.resolution, K).mean(axis=1)
            hypnodensity_30s = contents["predicted"]
            if len(hypnodensity) == 0:
                continue
            # if len(contents) == 0:
            #     continue
            # hypnodensity = softmax(contents)

            features[:, i] = extract_features.extract(hypnodensity, hypnodensity_30s)

        # Save raw features
        # saveFile = os.path.join(narco_feature_path, f"{f}_r{args.resolution:02}_trainD_unscaled.csv")
        saveFile = os.path.join(narco_feature_path, f"{model_str}_r{args.resolution:02}_trainD_unscaled.csv")
        print("{} | Saving {}".format(datetime.now(), saveFile))
        data_df = pd.DataFrame({"ID": ID, "Cohort": cohort, "Label": labels}).join(pd.DataFrame(features.T))
        data_df.to_csv(saveFile)

        # Scale data and save
        pct15_85 = (
            np.expand_dims(np.nanpercentile(features, 85, axis=1) - np.nanpercentile(features, 15, axis=1), axis=1)
            + 1e-5
        )
        pct50 = np.expand_dims(np.nanpercentile(features, 50, axis=1), axis=1)
        features = (features - pct50) / pct15_85
        #         features[features > 10] = 10
        #         features[features < -10] = -10

        # saveFile = os.path.join(narco_feature_path, f'{f}_r{args.resolution:02}_trainD.p')
        # saveFile = os.path.join(narco_feature_path, f"{f}_r{args.resolution:02}_trainD.csv")
        saveFile = os.path.join(narco_feature_path, f"{model_str}_r{args.resolution:02}_trainD.csv")
        print("{} | Saving {}".format(datetime.now(), saveFile))
        scaled_data_df = pd.DataFrame({"ID": ID, "Cohort": cohort, "Label": labels}).join(pd.DataFrame(features.T))
        scaled_data_df.to_csv(saveFile)

        scaleFile = os.path.join(narco_scaling_path, f"{model_str}_r{args.resolution:02}_scale.csv")
        print("{} | Saving {}".format(datetime.now(), scaleFile))
        scale_df = pd.DataFrame({"meanV": pct50.squeeze(), "scaleV": pct15_85.squeeze()})
        scale_df.to_csv(scaleFile)


# with open(saveFile, 'wb') as fi:
#     D = {'features': features, 'labels': labels, 'ID': ID}
#     pickle.dump(D, fi)

# F = pd.DataFrame(np.transpose(features))

# if count == 1:
#     variables = calculate_vif(F)
#     with open(os.path.join(narcoFeatureSelectPath, 'narcoFeatureSelect.p'), 'wb') as p:
#         pickle.dump(variables, p)
# sel_features = run_selection(np.transpose(features),DF['label'].values)
# break
# featureImportanceList[count] = sel_features


#         scale_df = pd.DataFrame({'meanV': pct50.squeeze(), 'scaleV': pct15_85.squeeze()})
#         scale_df.to_csv(os.path.join(narco_scaling_path, f'{f}_r{args.resolution:02}_scale.csv'))


# scale = {'meanV': pct50,
#          'scaleV': pct15_85}

# with open(os.path.join(narco_scaling_path, f + '_scale.p'), 'wb') as fi:
#     print('{} | Saving feature scaling for {}'.format(datetime.now(), f))
#     pickle.dump(scale, fi)

# ID, ID_ling, labelT, trainLT, df = load_csv()

# # P = '/home/jens/Documents/stanford/scored_data/'
# # D = os.listdir(P)
# P = './data/massc'
# D = os.listdir(P)
# # D = glob(os.path.join(P, '**', '*.pkl'))
# D.sort()


# for d in D:
#     # F = os.listdir(P+d)
#     F = glob(os.path.join(P, d, '**', '*.pkl'))
#     F.sort()
#     count = 0
#     labels = np.zeros(len(F))
#     trainL = np.zeros(len(F))
#     featStack = []

#     for idx, f in enumerate(tqdm(F)):
#         if idx < 100:
#             continue
#         # name = f[30:-7]
#         name = os.path.basename(f)[6:-4]
#         if 'notte' in name:
#             name = name[:5]
#         elif (len(name) == 8) & (name[0] == '0'):
#             name = name[1:]
#         try:
#             index = ID.index(name)
#         except:
#             try:
#                 index = ID_ling.index(name)
#             except:
#                 print(name + ' not found')
#                 continue

#         # print(str(count) + '/' + str(len(F)))
#         # contents = sio.loadmat(P+d+'/'+f)
#         # pred = contents['predictions']
#         with open(f, 'rb') as fp:
#             pred = pickle.load(fp)['yhat_15s']

#         if len(pred)==0:
#             continue

#         labels[count] = labelT[index]
#         trainL[count] = trainLT[index]

#         feat = extract_features.extract(pred)
#         feat = np.expand_dims(feat,axis=1)
#         if len(featStack)==0:
#             featStack = feat
#         else:
#             featStack = np.concatenate([featStack,feat],axis=1)

#         count += 1
#     featStack = np.transpose(featStack)
#     labels = labels[:count]
#     trainL = trainL[:count]
#     m = np.mean(featStack,axis=1)+1e-10
#     v = np.percentile(featStack,85,axis=1) - np.percentile(featStack,15,axis=1)+1e-10

#     m = np.expand_dims(m,axis=1)
#     featStackScaled = featStack/np.expand_dims(v,axis=1)

#     featStackScaled[10<featStackScaled] = 10
#     featStackScaled[-10>featStackScaled] = -10

#     data = {'features': featStackScaled,
#             'labels': labels,
#             'trainL': trainL,
#             }

#     scale = {'mean': m,
#              'range': v}

#     output = open('narco_features/' + d +'_narcodata.pkl', 'wb')
#     pickle.dump(data, output, -1)
#     output.close()
