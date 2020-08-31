from scipy.io import loadmat
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    """
    Return data in np format
    """

    f1 = 'Datasets/PET_India.mat'
    f2 = 'Datasets/rain_India.mat'
    return(loadmat(f1)['PET'], loadmat(f2)['rain'])


def yearly():
    """
    Convert to yearly data
    """

    MPET = np.zeros(121*121*64).reshape((121, 121, 64))
    MR = np.zeros(121*121*64).reshape((121, 121, 64))
    for i in range(64):
        t1 = PET[:, :, np.arange(12)+12*i]
        t2 = np.sum(t1, axis=2)
        MPET[:, :, i] = t2

        t1 = R[:, :, np.arange(12)+12*i]
        t2 = np.sum(t1, axis=2)
        MR[:, :, i] = t2

    return MPET, MR


def get_mask():
    """
    Create mask file for changing grid points
    """

    M = np.zeros(121*121*64).reshape((121, 121, 64))
    MPET, MR = yearly()
    for i in range(64):
        t1 = MPET[:, :, i]-MR[:, :, i]
        t1[t1 < 0] = -1  # energy limited
        t1[t1 > 0] = 1  # water limited
        M[:, :, i] = t1

    mask = np.ones(121*121).reshape((121, 121))

    for i in range(121):
        for j in range(121):
            # if np.unique(M[i][j]).size > 1 or not np.all(M[i][j]):
            if np.unique(M[i][j]).size > 1:
                mask[i][j] = 0

    mask = mask.transpose()
    mask = np.flipud(mask)
    return mask


def plot_hm(X):
    """
    Plot heat map
    """

    X = X.transpose()
    X = np.flipud(X)
    mask = get_mask()
    ax = sns.heatmap(X, xticklabels=False, yticklabels=False,
                     cmap="YlGnBu", vmin=-5, vmax=5, mask=mask)
    plt.title("SDI for changing grid points")
    plt.show()


def calSDI():
    """
    Returns SDI for R and PET data
    """

    M = np.zeros(121*121).reshape((121, 121))
    MPET, MR = yearly()

    for i in range(121):
        for j in range(121):

            SurINT = MR[i][j]-MPET[i][j]
            SurINT = SurINT.flatten()
            SurINT[SurINT < 0] = 0
            # SurINT[SurINT!=0]

            DT = MPET[i][j]-MR[i][j]
            DT = DT.flatten()
            DT[DT < 0] = 0
            # DT[DT!=0]

            SDI = (preprocessing.scale(SurINT)+preprocessing.scale(DT)) / \
                (np.std(preprocessing.scale(SurINT))+preprocessing.scale(DT))
            SDI = np.mean(SDI)
            M[i][j] = SDI

    plot_hm(M)


if __name__ == "__main__":

    PET, R = load_data()
    # (121, 121, 768) 64 years monthly data

    calSDI()
