from scipy.io import loadmat
import numpy as np
from sklearn import preprocessing


def load_data():
    """
    Return data in np format
    """

    f1 = 'Datasets/PET_India.mat'
    f2 = 'Datasets/rain_India.mat'
    return(loadmat(f1)['PET'], loadmat(f2)['rain'])



def calSDI():
    """
    Returns SDI for R and PET data
    """

    SurINT=R-PET
    SurINT=SurINT.flatten()
    SurINT[SurINT<0]=0
    # SurINT[SurINT!=0]

    DT=PET-R
    DT=DT.flatten()
    DT[DT<0]=0
    # DT[DT!=0]

    SDI=(preprocessing.scale(SurINT)+preprocessing.scale(DT))/(np.std(preprocessing.scale(SurINT))+preprocessing.scale(DT))
    # print(SDI.shape)



if __name__ == "__main__":

    PET, R = load_data()
    # (121, 121, 768) 64 years monthly data

    calSDI()