import itertools

import numpy as np
import pywt

from utils.rolling_window import rolling_window_nodelay


def extract_features(hyp, resolution, hyp_30s=None):

    eps = 1e-10
    features = np.zeros([24 + 31 * 15])
    j = -1
    f = 10

    if not np.allclose(hyp.sum(axis=1)[0], 1.0):
        print("Softmax'ing")
        hyp = softmax(hyp)

    if hyp_30s is not None:
        if not np.allclose(hyp.sum(axis=1)[0], 1.0):
            print("Softmax'ing")
            hyp = softmax(hyp)

    for i in range(5):
        for comb in itertools.combinations([0, 1, 2, 3, 4], i + 1):
            j += 1
            dat = np.prod(hyp[:, comb], axis=1) ** (1 / float(len(comb)))

            dat_mean = np.mean(dat)
            dat_max = np.max(dat)
            dat_diff = np.diff(dat)
            dat_waventr = wavelet_entropy(dat)

            features[j * 15] = np.log(dat_mean + eps)
            features[j * 15 + 1] = -np.log(1 - dat_max + eps)

            # moving_av = np.convolve(dat, np.ones(self.moving_avg_epochs), mode="valid")
            # features[j * 15 + 2] = np.mean(np.abs(np.diff(moving_av)))
            features[j * 15 + 2] = np.mean(np.abs(dat_diff))

            features[j * 15 + 3] = dat_waventr  # Shannon entropy - check if it is used as a feature

            rate = np.cumsum(dat) / np.sum(dat)
            I1 = next(i for i, v in enumerate(rate) if v > 0.05)
            features[j * 15 + 4] = np.log(I1 * 2 + eps)
            I2 = next(i for i, v in enumerate(rate) if v > 0.1)
            features[j * 15 + 5] = np.log(I2 * 2 + eps)
            I3 = next(i for i, v in enumerate(rate) if v > 0.3)
            features[j * 15 + 6] = np.log(I3 * 2 + eps)
            I4 = next(i for i, v in enumerate(rate) if v > 0.5)
            features[j * 15 + 7] = np.log(I4 * 2 + eps)

            features[j * 15 + 8] = np.sqrt(dat_max * dat_mean + eps)
            features[j * 15 + 9] = np.mean(np.abs(dat_diff) * dat_mean + eps)
            features[j * 15 + 10] = np.log(dat_waventr * dat_mean + eps)
            features[j * 15 + 11] = np.sqrt(I1 * 2 * dat_mean)
            features[j * 15 + 12] = np.sqrt(I2 * 2 * dat_mean)
            features[j * 15 + 13] = np.sqrt(I3 * 2 * dat_mean)
            features[j * 15 + 14] = np.sqrt(I4 * 2 * dat_mean)

    if hyp_30s is not None:
        # Use 30 s scoring for the next features
        data = hyp_30s
    else:
        rem = hyp.shape[0] % int(30 // resolution)
        if rem > 0:
            data = hyp[:-rem, :]
        else:
            data = hyp

        data = data.reshape([-1, int(30 // resolution), 5])
        data = np.squeeze(np.mean(data, axis=1))

    S = np.argmax(data, axis=1)

    SL = [i for i, v in enumerate(S) if v != 0]
    if len(SL) == 0:
        SL = len(data)
    else:
        SL = SL[0]

    RL = [i for i, v in enumerate(S) if v == 4]
    if len(RL) == 0:
        RL = len(data)
    else:
        RL = RL[0]

    # Nightly SOREMP
    wCount = 0
    rCount = 0
    rCountR = 0
    soremC = 0
    """
    # The following was originally used, but found to be inconsistent with the described feature it implements.
    for i in range(SL, len(S)):
        if (S[i] == 0) | (S[i] == 1):
            wCount += 1
        elif (S[i] == 4) & (wCount > 4):
            rCount += 1
            rCountR += 1
        elif rCount > 1:
            soremC += 1
        else:
            wCount = 0
            rCount = 0
    """

    """
    Updated
    This ensures we meet the criteria for a SOREMP and also takes care of counting the first epoch of REM of
    that SOREMP.  The manuscript code took care of the first epoch of REM but used too general of a description
    for a SOREMP (i.e. missed the minimum requirement of one minute of REM).
    """
    for i in range(SL, len(S)):
        if (S[i] == 0) | (S[i] == 1):
            wCount += 1
        elif (S[i] == 4) & (wCount > 4):
            rCount += 1
            if rCount == 2:
                soremC += 1
                rCountR += 2
            elif rCount > 2:
                rCountR += 1
        else:
            wCount = 0
            rCount = 0

    # NREM Fragmentation
    nCount = 0
    nFrag = 0
    for i in range(SL, len(S)):
        if (S[i] == 2) | (S[i] == 3):
            nCount += 1
        elif ((S[i] == 0) | (S[i] == 1)) & (nCount > 3):
            nFrag += 1
            nCount = 0

    # W/N1 Bouts
    wCount = 0
    wBout = 0
    wCum = 0
    sCount = 0
    for i in range(SL, len(S)):
        if S[i] != 1:
            sCount += 1

        if (sCount > 5) & ((S[i] == 0) | (S[i] == 1)):
            wCount = wCount + 1
            if wCount < 30:
                wCum = wCum + 1

        elif wCount > 4:
            wCount = 0
            wBout = wBout + 1

    #

    features[-24] = logmodulus(SL * f)
    features[-23] = logmodulus((RL - SL) * f)

    features[-22] = np.sqrt(rCountR)
    features[-21] = np.sqrt(soremC)
    features[-20] = np.sqrt(nFrag)
    features[-19] = np.sqrt(wCum)
    features[-18] = np.sqrt(wBout)

    # Find out what features are used:...!
    data = hyp
    features[-17:] = logmodulus(transitionFeatures(data))

    return features


def logmodulus(x):
    return np.sign(x) * np.log(abs(x) + 1)


def transitionFeatures(data):
    # S = np.zeros(data.shape)
    # for i in range(5):
    #     S[:, i] = np.convolve(data[:, i], np.ones(9), mode="same")

    # # if not np.allclose(S.sum(axis=1)[0], 1.0):
    # #     print("Softmax'ing")
    # S = self.softmax(S)

    S = data

    cumR = np.zeros(S.shape)
    Th = 0.2
    peakTh = 10
    for j in range(5):
        for i in range(len(S)):
            if S[i - 1, j] > Th:
                cumR[i, j] = cumR[i - 1, j] + S[i - 1, j]

        cumR[cumR[:, j] < peakTh, j] = 0

    for i in range(5):
        d = cumR[:, i]
        indP = find_peaks(cumR[:, i])
        typeP = np.ones(len(indP)) * i
        if i == 0:
            peaks = np.concatenate([np.expand_dims(indP, axis=1), np.expand_dims(typeP, axis=1)], axis=1)
        else:
            peaks = np.concatenate(
                [peaks, np.concatenate([np.expand_dims(indP, axis=1), np.expand_dims(typeP, axis=1)], axis=1)], axis=0,
            )

    I = [i[0] for i in sorted(enumerate(peaks[:, 0]), key=lambda x: x[1])]
    peaks = peaks[I, :]

    remList = np.zeros(len(peaks))

    # This merges W and N1 into one category
    peaks[peaks[:, 1] == 0, 1] = 1
    peaks[:, 1] = peaks[:, 1] - 1

    if peaks.shape[0] < 2:
        features = np.zeros(17)
        return features

    for i in range(peaks.shape[0] - 1):
        if peaks[i, 1] == peaks[i + 1, 1]:
            peaks[i + 1, 0] += peaks[i, 0]
            remList[i] = 1
    remList = remList == 0
    peaks = peaks[remList, :]
    transitions = np.zeros([4, 4])

    for i in range(peaks.shape[0] - 1):
        transitions[int(peaks[i, 1]), int(peaks[i + 1, 1])] = np.sqrt(peaks[i, 0] * peaks[i + 1, 0])
    di = np.diag_indices(4)
    transitions[di] = None

    transitions = transitions.reshape(-1)
    transitions = transitions[np.invert(np.isnan(transitions))]
    nPeaks = np.zeros(5)
    for i in range(4):
        nPeaks[i] = np.sum(peaks[:, 1] == i)

    nPeaks[-1] = peaks.shape[0]

    features = np.concatenate([transitions, nPeaks], axis=0)
    return features


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    div = np.repeat(np.expand_dims(np.sum(e_x, axis=1), 1), 5, axis=1)
    return np.divide(e_x, div)


def find_peaks(x):
    peaks = []
    for i in range(1, len(x)):
        if x[i - 1] > x[i]:
            peaks.append(i - 1)

    return np.asarray(peaks)


def wavelet_entropy(dat):
    coef, _ = pywt.cwt(dat, np.arange(1, 60), "gaus1", method="fft")
    Eai = np.sum(np.square(np.abs(coef)), axis=1)
    pai = Eai / np.sum(Eai)

    WE = -np.sum(np.log(pai) * pai)

    return WE


def calc_sorem(hyp, resolution, sorem_thresholds):

    eps = 1e-10
    #     features = np.zeros([24 + 31 * 15])
    features = []
    j = -1
    f = 10

    if not np.allclose(hyp.sum(axis=1)[0], 1.0):
        print("Softmax'ing")
        hyp = softmax(hyp)
    data = hyp
    S = np.argmax(data, axis=1)

    # Sleep latency
    SL = [i for i, v in enumerate(S) if v != 0]
    if len(SL) == 0:
        SL = len(data)
    else:
        SL = SL[0]

    # REM latency (SL will be subtracted later)
    RL = [i for i, v in enumerate(S) if v == 4]
    if len(RL) == 0:
        RL = len(data)
    else:
        RL = RL[0]

    def soremp(S, SL, threshold=4, rcount_threshold=2):
        wCount = 0
        rCount = 0
        rCountR = 0
        soremC = 0
        """
        # The following was originally used, but found to be inconsistent with the described feature it implements.
        for i in range(SL, len(S)):
            if (S[i] == 0) | (S[i] == 1):
                wCount += 1
            elif (S[i] == 4) & (wCount > 4):
                rCount += 1
                rCountR += 1
            elif rCount > 1:
                soremC += 1
            else:
                wCount = 0
                rCount = 0
        """

        """
        Updated
        This ensures we meet the criteria for a SOREMP and also takes care of counting the first epoch of REM of
        that SOREMP.  The manuscript code took care of the first epoch of REM but used too general of a description
        for a SOREMP (i.e. missed the minimum requirement of one minute of REM).
        """
        for i in range(SL, len(S)):
            if (S[i] == 0) | (S[i] == 1):
                wCount += 1
            elif (S[i] == 4) & (wCount > threshold):
                rCount += 1
                if rCount == rcount_threshold:
                    soremC += 1
                    rCountR += rcount_threshold
                elif rCount > rcount_threshold:
                    rCountR += 1
            else:
                wCount = 0
                rCount = 0
        return rCountR, soremC

    for thr in sorem_thresholds:
        rCountR, soremC = soremp(S, SL, threshold=thr)
        features.append((thr, rCountR, soremC))

    return features


def calc_nremfrag(hyp, resolution, nremfrag_thresholds):

    eps = 1e-10
    #     features = np.zeros([24 + 31 * 15])
    features = []
    j = -1
    f = 10

    if not np.allclose(hyp.sum(axis=1)[0], 1.0):
        print("Softmax'ing")
        hyp = softmax(hyp)
    data = hyp
    S = np.argmax(data, axis=1)

    # Sleep latency
    SL = [i for i, v in enumerate(S) if v != 0]
    if len(SL) == 0:
        SL = len(data)
    else:
        SL = SL[0]

    # REM latency (SL will be subtracted later)
    RL = [i for i, v in enumerate(S) if v == 4]
    if len(RL) == 0:
        RL = len(data)
    else:
        RL = RL[0]

    # NREM Fragmentation
    def nrem_fragmentation(threshold=3):
        nCount = 0
        nFrag = 0
        for i in range(SL, len(S)):
            if (S[i] == 2) | (S[i] == 3):
                nCount += 1
            elif ((S[i] == 0) | (S[i] == 1)) & (nCount > threshold):
                nFrag += 1
                nCount = 0
        return nFrag

    for thr in nremfrag_thresholds:
        nFrag = nrem_fragmentation(threshold=thr)
        features.append((thr, nFrag))

    return features


def calc_transition_triples(hypnodensity, resolution):
    """Calculate the transition triplet frequencies per 1.5 hour time segments as described in
    Perslev, M., Darkner, S., Kempfner, L. et al. U-Sleep: resilient high-frequency sleep staging.
    npj Digit. Med. 4, 72 (2021). https://doi.org/10.1038/s41746-021-00440-5.

    Args:
        hypnodensity (array_like): A `N x 5` array containing sleep stage probabilities.
        resolution (int): Resolution of supplied hypnodensity. E.g. `resolution=3` means 1 prediction every 3 s.

    Returns:
        transition_triples: a set of transition triplet categories.
        features (array_like): A `K x 80` array containing bin counts for each triplet. Here, `K` is the number of 1.5 h segments in `hypnodensity`.
    """

    transition_triplets = set(
        [trip for trip in list(itertools.product(*[[0, 1, 2, 3, 4]] * 3)) if (trip[0] != trip[1] and trip[1] != trip[2])]
    )

    features = []

    if not np.allclose(hypnodensity.sum(axis=1)[0], 1.0):
        print("Softmax'ing")
        hypnodensity = softmax(hypnodensity)

    # Get hypnogram
    hypnogram = np.argmax(hypnodensity, axis=1)

    # Get how many 1.5 h segments
    segments_per_resolution = 1.5 * 3600 / resolution
    n_segments = int(len(hypnogram) // segments_per_resolution)

    # Split the hypnogram into 1.5 h chunks
    chunked_hypnogram = rolling_window_nodelay(hypnogram, int(segments_per_resolution), int(segments_per_resolution)).T
    chunked_hypnogram = chunked_hypnogram[:n_segments]

    # For each segment, calculate the distribution of transition triplets
    i = 0
    bin_counts = []
    for hypnogram_segment in chunked_hypnogram:
        bin_count = np.zeros(len(transition_triplets))
        while i < segments_per_resolution - 2:
            triplet_candidate = hypnogram_segment[i : i + 3]
            bin_idx = [trip_idx for trip_idx, trip in enumerate(transition_triplets) if tuple(triplet_candidate) == trip]
            if bin_idx:
                bin_count[bin_idx] += 1
            i += 1
        bin_counts.append(bin_count)
    features = np.asarray(bin_counts)

    return transition_triplets, features


def multi_step_transition_matrix(x, num_steps=8, resolution=None):
    """Calculates the transition matrix at multiple logaritmic steps for x.

    @author Mads Olsen

    :param x: input 2D matrix, with x.shape[0]: timesteps and x.shape[1]: num_classes.
    :param num_steps: number of matrix multiplications calculated at logarithmically spaced steps.
        x.shape[0] >= 2 ** num_steps
    :return: Multi-step transition matrix with shape: [num_steps * num_classes ** 2]

    """
    # Per correspondence w. Mads Olsen
    # This should be changed to automatically adjust based on the resolution
    if resolution is not None:
        num_steps = int(np.ceil(np.log2(7200 / resolution)))
        # if resolution == 128:
        #     num_steps = 20
        # elif resolution == 30:
        #     num_steps = 8

    step_idx = [2 ** idx for idx in range(num_steps)]
    matmulfun = lambda tau: np.matmul(x[:-tau, :].transpose(), x[tau:, :]) / x[:-tau, :].shape[0]
    return np.array(list(map(matmulfun, step_idx))).flatten()


if __name__ == "__main__":
    hypnogram_duration_h = 8
    resolution = 5
    hypnogram_dur_s = 8 * 3600 // resolution
    hypnodensity = softmax(np.random.randn(hypnogram_dur_s, 5))

    features = calc_transition_triples(hypnodensity, resolution)
