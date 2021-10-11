import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve


def plot_roc(y_true, y_pred, thr_val=None, title=None, savepath=None):
    fpr, tpr, thr = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(
        fpr, tpr, color="orange", lw=lw, label="{} (AUC = {:.3})".format(title, roc_auc),
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw / 4, linestyle="--")
    if not isinstance(thr, list):
        thr_idx = np.argmax(thr <= thr_val)
        plt.plot(fpr[thr_idx], tpr[thr_idx], color="navy", marker="o")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("1 - specificity")
    plt.ylabel("Sensitivity")
    plt.title("Receiver operating characteristic curve")
    plt.legend(loc="lower right")

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")


def plot_roc_ensemble(y_true, y_pred, thr_val=None, title=None, figtitle=None, savepath=None):
    assert (isinstance(y_true, list) and isinstance(y_pred, list)) and (
        len(y_true) == len(y_pred)
    ), "Arguments y_true, y_pred should be lists of equal length"

    lw = 1
    plt.figure()
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    for y_t, y_p, t in zip(y_true[:-1], y_pred[:-1], title[:-1]):
        fpr, tpr, thr = roc_curve(y_t, y_p)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, ":", lw=lw, label=f"{t}, AUC={roc_auc:.3f}")
        if not isinstance(thr, list):
            thr_idx = np.argmax(thr < thr_val)
            plt.plot(fpr[thr_idx], tpr[thr_idx], color="navy", marker="o", markersize=2)
    # Run over the last exp, which is assumed to be the ensemble
    fpr, tpr, thr = roc_curve(y_true[-1], y_pred[-1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, label=f"{title[-1]}, AUC={roc_auc:.3f})")
    if not isinstance(thr, list):
        thr_idx = np.argmax(thr < thr_val)
        plt.plot(fpr[thr_idx], tpr[thr_idx], color="navy", marker="o", markersize=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    # plt.gca().set_aspect("equal")
    plt.xlabel("1 - specificity")
    plt.ylabel("Sensitivity")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.title(figtitle or "Receiver operating characteristic curve")
    plt.grid(color="0.5", linestyle="--", linewidth=0.5)
    plt.xticks(ticks=np.linspace(0.0, 1.0, 11, endpoint=True))
    plt.yticks(ticks=np.linspace(0.0, 1.0, 11, endpoint=True))
    plt.legend(loc="lower right")

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches="tight")


def plot_hypnodensity(logits, preds, trues, title=None, save_path=None):

    # Setup title
    f, ax = plt.subplots(nrows=4, figsize=(20, 5), dpi=400)
    f.suptitle(title)

    # Setup colors
    cmap = np.array(
        [
            [0.4353, 0.8157, 0.9353],  # W
            [0.9490, 0.9333, 0.7725],  # N1
            [0.9490, 0.6078, 0.5118],  # N2
            [0.6863, 0.2078, 0.2784],  # N3
            [0.0000, 0.4549, 0.7373],
        ],  # R
    )

    # Plot the hypnodensity
    h = logits.T
    hypnodensity = np.concatenate([h, h[:, -1, np.newaxis]], axis=-1)
    y_ = np.zeros((hypnodensity.shape[0] + 1, hypnodensity.shape[1]))
    y_[1:, :] = np.cumsum(hypnodensity, axis=0)
    l = []
    for n in range(hypnodensity.shape[0]):
        l.append(
            ax[0].fill_between(
                np.arange(hypnodensity.shape[1]),
                y_[n, :],
                y_[n + 1, :],
                edgecolor="face",
                facecolor=cmap[n, :],
                linewidth=0.0,
                step="post",
            )
        )
    ax[0].get_xaxis().set_visible(False)
    ax[0].set_xlim(0, hypnodensity.shape[1] - 1)
    ax[0].set_ylim(0.0, 1.0)
    ax[0].set_ylabel("1 s")
    plt.setp(ax[0].get_yticklabels(), visible=False)
    ax[0].tick_params(axis="both", which="both", length=0)

    # Create legend
    legend_elements = [
        mpl.patches.Patch(facecolor=cm, edgecolor=cm, label=lbl) for cm, lbl in zip(cmap, ["W", "N1", "N2", "N3", "REM"])
    ]
    ax[0].legend(handles=legend_elements, loc="lower center", bbox_to_anchor=[0.5, 1.0], ncol=5)
    #     sns.despine(top=True, bottom=True, left=True, right=True)
    #     plt.tight_layout()

    # Plot predicted hypnodensity at 30 s
    h = preds.T
    hypnodensity = np.concatenate([h, h[:, -1, np.newaxis]], axis=-1)
    y_ = np.zeros((hypnodensity.shape[0] + 1, hypnodensity.shape[1]))
    y_[1:, :] = np.cumsum(hypnodensity, axis=0)
    l = []
    for n in range(hypnodensity.shape[0]):
        l.append(
            ax[1].fill_between(
                np.arange(hypnodensity.shape[1]),
                y_[n, :],
                y_[n + 1, :],
                edgecolor="face",
                facecolor=cmap[n, :],
                linewidth=0.0,
                step="post",
            )
        )
    ax[1].get_xaxis().set_visible(False)
    ax[1].set_xlim(0, hypnodensity.shape[1] - 1)
    ax[1].set_ylim(0.0, 1.0)
    ax[1].set_ylabel("30 s")
    plt.setp(ax[1].get_yticklabels(), visible=False)
    ax[1].tick_params(axis="both", which="both", length=0)

    # Plot predicted hyponogram
    ax[2].plot(preds.argmax(axis=-1))
    ax[2].set_xlim(0, trues.shape[0] - 1)
    ax[2].set_ylim(-0.5, 4.5)
    ax[2].get_xaxis().set_visible(False)
    ax[2].set_yticks([0, 1, 2, 3, 4])
    ax[2].set_yticklabels(["W", "N1", "N2", "N3", "R"])
    ax[2].set_ylabel("Automatic")

    # Plot true hyponogram
    ax[3].plot(trues.argmax(axis=-1))
    ax[3].set_xlim(0, trues.shape[0] - 1)
    ax[3].set_ylim(-0.5, 4.5)
    ax[3].set_yticks([0, 1, 2, 3, 4])
    ax[3].set_yticklabels(["W", "N1", "N2", "N3", "R"])
    ax[3].set_xticks(np.arange(0, trues.shape[0] - 1, 20))
    ax[3].set_xticklabels(np.arange(0, trues.shape[0] - 1, 20) * 30 // 60)
    ax[3].set_xlabel("Time (min)")
    ax[3].set_ylabel("Manual")

    # Save figure
    if save_path is not None:
        f.savefig(f"results/{save_path}", dpi=300, bbox_inches="tight", pad_inches=0)
    #     plt.close()
    plt.show()
