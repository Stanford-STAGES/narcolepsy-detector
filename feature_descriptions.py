import itertools

import numpy as np
import pandas as pd


def get_feature_descriptions(selected_features):

    feature_type_descriptions = [
        "Log average",
        "Log maximum",
        "Log average differential",
        "Log Shannon entropy of wavelet coeffs.",
        "Log time until 5% maximum",
        "Log time until 10% maximum",
        "Log time until 30% maximum",
        "Log time until 50% maximum",
        "Weighted maximum value",
        "Weighted average differential",
        "Weighted log Shannon entropy",
        "Weighted time until 5% maximum",
        "Weighted time until 10% maximum",
        "Weighted time until 30% maximum",
        "Weighted time until 50% maximum",
    ]

    feature_descriptions = []

    j = -1
    for i in range(5):
        for comb in itertools.combinations(["W", "N1", "N2", "N3", "R"], i + 1):
            j += 1
            for f in range(15):
                feature_descriptions.append((str(f + 1), comb, feature_type_descriptions[f]))

            # feature_descriptions
            # features[j*15] = np.log(np.mean(dat) + eps)
            # features[j*15+1] = -np.log(1-np.max(dat) + eps)

            # moving_av = np.convolve(dat,np.ones(10),mode='valid')
            # features[j*15+2] = np.mean(np.abs(np.diff(moving_av)))

            # features[j*15+3] = wavelet_entropy(dat) #Shannon entropy - check if it is used as a feature

            # rate = np.cumsum(dat)/np.sum(dat)
            # I1 = next(i for i,v in enumerate(rate) if v>0.05)
            # features[j*15+4] = np.log(I1*2+eps)
            # I2 = next(i for i,v in enumerate(rate) if v>0.1)
            # features[j*15+5] = np.log(I2*2+eps)
            # I3 = next(i for i,v in enumerate(rate) if v>0.3)
            # features[j*15+6] = np.log(I3*2+eps)
            # I4 = next(i for i,v in enumerate(rate) if v>0.5)
            # features[j*15+7] = np.log(I4*2+eps)

            # features[j*15+8] = np.sqrt(np.max(dat)*np.mean(dat)+eps)
            # features[j*15+9] = np.mean(np.abs(np.diff(dat))*np.mean(dat)+eps)
            # features[j*15+10] = np.log(wavelet_entropy(dat)*np.mean(dat)+eps)
            # features[j*15+11] = np.sqrt(I1*2*np.mean(dat))
            # features[j*15+12] = np.sqrt(I2*2*np.mean(dat))
            # features[j*15+13] = np.sqrt(I3*2*np.mean(dat))
            # features[j*15+14] = np.sqrt(I4*2*np.mean(dat))

    feature_descriptions.append(("SL"))
    feature_descriptions.append(("REML"))
    feature_descriptions.append(("cumulative REM duration following W/N1 periods > 2.5 min"))
    feature_descriptions.append(("total nightly SOREMP duration following W/N1 periods > 2.5 min"))
    feature_descriptions.append(("NREM fragmentation"))
    feature_descriptions.append(("Cumulative W/N1 duration for periods < 15 min"))
    feature_descriptions.append(("Number of W/N1 bouts"))
    feature_descriptions.append(("Transition: W/N1 -> N2"))
    feature_descriptions.append(("Transition: W/N1 -> N3"))
    feature_descriptions.append(("Transition: W/N1 -> R"))
    feature_descriptions.append(("Transition: N2 -> W/N1"))
    feature_descriptions.append(("Transition: N2 -> N3"))
    feature_descriptions.append(("Transition: N2 -> R"))
    feature_descriptions.append(("Transition: N3 -> W/N1"))
    feature_descriptions.append(("Transition: N3 -> N1"))
    feature_descriptions.append(("Transition: N3 -> R"))
    feature_descriptions.append(("Transition: R -> W/N1"))
    feature_descriptions.append(("Transition: R -> N2"))
    feature_descriptions.append(("Transition: R -> N3"))
    feature_descriptions.append(("Transition peak height: W/N1"))
    feature_descriptions.append(("Transition peak height: N2"))
    feature_descriptions.append(("Transition peak height: N3"))
    feature_descriptions.append(("Transition peak height: R"))
    feature_descriptions.append(("Transition peak height: total"))
    # for i in range(17):
    #     feature_descriptions.append((f"Transition feature #{i+1}"))

    #     print(len(feature_descriptions))
    output = []
    for idx, f in enumerate(selected_features):
        #         print(idx, f, feature_descriptions[f])
        output.append((idx, f, feature_descriptions[f]))

    return pd.DataFrame(
        {
            "FeatureIdx": [f[1] for f in output],
            "FeatureTypeIdx": [f[-1][0] if isinstance(f[-1], tuple) else np.nan for f in output],
            "StageCombination": [f[-1][1] if isinstance(f[-1], tuple) else np.nan for f in output],
            "FeatureDescription": [f[-1][-1] if isinstance(f[-1], tuple) else f[-1] for f in output],
        }
    )

    # features[-24] = SL*f  # Sleep latency
    # features[-23] = RL-SL*f  # REM latency

    # features[-22] = rCountR  # cumulative REM duration following W/N1 periods > 2.5 min
    # features[-21] = soremC  # total nightly SOREMP duration following W/N1 periods > 2.5 min
    # features[-20] = nFrag  # NREM fragmentation
    # features[-19] = wCum  # Cumulative W/N1 duration for periods < 15 min
    # features[-18] = wBout  # number of W/N1 bouts


if __name__ == "__main__":

    # fmt: off
    # selected_features = (1, 11, 16, 22, 25, 41, 43, 49, 64, 65, 86, 87, 103, 119, 140, 147, 149, 166, 196, 201, 202, 220, 244, 245, 261, 276, 289, 296, 299, 390, 405, 450, 467, 468, 470, 474, 476, 477)
    # selected_features = (1, 11, 16, 22, 25, 41, 43, 49, 64, 65, 86, 87, 103, 119, 140, 147, 149, 166, 196, 201, 202, 220, 244, 245, 261, 276, 289, 296, 299, 390, 405, 450, 467, 468, 470)
    # selected_features = (1, 4, 11, 16, 22, 25, 41, 43, 49, 64, 65, 86, 87, 103, 119, 140, 147, 149, 166, 196, 201, 202, 220, 244, 245, 261, 276, 289, 296, 390, 405, 450, 467, 468, 470)
    selected_features = range(489)
    # fmt: on

    feature_descriptions = get_feature_descriptions(selected_features)
    print("")
    print(
        "| Hypnodensity-derived features                                                                                                   |"
    )
    print(
        "|=================================================================================================================================|"
    )
    print(feature_descriptions.to_markdown(index=False))
