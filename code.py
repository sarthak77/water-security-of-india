import numpy as np
import matplotlib.pyplot as plt



def load_data():
    """
    Return data in np format
    """

    from scipy.io import loadmat
    f1='Datasets/PET_India.mat'
    f2='Datasets/rain_India.mat'
    return(loadmat(f1)['PET'],loadmat(f2)['rain'])



if __name__ == "__main__":
    
    PET,R=load_data()
    print(PET.shape)
    print(R.shape)
    # (121, 121, 768) 64 years monthly data