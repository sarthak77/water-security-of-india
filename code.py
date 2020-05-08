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



def plot_hm(X,T):
    """
    Plot heat map
    """

    import seaborn as sns
    X=X.transpose()
    X=np.flipud(X)
    # fig=plt.figure()
    ax=sns.heatmap(X,xticklabels=False,yticklabels=False,cmap="YlGnBu",cbar=False)
    plt.title(T)
    # plt.show()
    # plt.savefig(T+".png")
    plt.savefig("img/j.png")
    # plt.close()



def monthly():
    """
    Convert to monthly data
    """

    MPET=np.zeros(121*121*64).reshape((121,121,64))
    MR=np.zeros(121*121*64).reshape((121,121,64))
    for i in range(64):
        t1=PET[:,:,np.arange(12)+12*i]
        t2=np.sum(t1,axis=2)
        MPET[:,:,i]=t2

        t1=R[:,:,np.arange(12)+12*i]
        t2=np.sum(t1,axis=2)
        MR[:,:,i]=t2

    return MPET,MR



def part1():
    """
    Yearly zone variation
    """

    MPET,MR=monthly()
    for i in range(1):
        t1=MPET[:,:,i]-MR[:,:,i]
        t1[t1<0]=-1#energy limited
        t1[t1>0]=1#water limited
        T="Water-Energy Zones For Year "+str(1951+i)
        plot_hm(t1,T)
    pass



if __name__ == "__main__":
    
    PET,R=load_data()
    print(PET.shape)
    print(R.shape)
    # (121, 121, 768) 64 years monthly data

    part1()