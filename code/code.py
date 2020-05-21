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



def get_mask():
    """
    Create mask file for changing grid points
    """

    M=np.zeros(121*121*64).reshape((121,121,64))
    MPET,MR=yearly()
    for i in range(64):
        t1=MPET[:,:,i]-MR[:,:,i]
        t1[t1<0]=-1#energy limited
        t1[t1>0]=1#water limited
        M[:,:,i]=t1

    mask=np.ones(121*121).reshape((121,121))

    for i in range(121):
        for j in range(121):
            if np.unique(M[i][j]).size > 1:
                mask[i][j]=0

    mask=mask.transpose()
    mask=np.flipud(mask)
    return mask



def zone_mask(zone):
    """
    Get mask of individual zones
    0 = sea zone
    1 = central zone
    2 = north-east zone
    3 = north-east-hills zone
    4 = south zone
    5 = J&K zone
    6 = west  zone
    7 = north zone
    """

    with open('./mask/homogeneouszones.txt', 'r') as file:
        input_lines = [line.strip("\n") for line in file]

    temp=[]
    for l in input_lines:
        a=[]
        l=l.replace(" ","")
        for i in l:
            a.append(int(i))
        temp.append(a)

    temp=np.array(temp)
    temp=temp == zone
    temp=np.invert(temp)
    return temp



def plot_hm(X,T):
    """
    Plot heat map
    """

    import seaborn as sns

    X=X.transpose()
    X=np.flipud(X)
    # mask=get_mask()#for changing grid points
    mask=zone_mask(7)#for individual zones
    ax=sns.heatmap(X,xticklabels=False,yticklabels=False,cmap="YlGnBu",cbar=False,mask=mask)
    # ax=sns.heatmap(X,xticklabels=False,yticklabels=False,cmap="YlGnBu",cbar=False)
    
    plt.title(T)
    # plt.show()
    plt.savefig("images/zone/7/"+T+".png")



def yearly():
    """
    Convert to yearly data
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

    MPET,MR=yearly()
    for i in range(64):
        t1=MPET[:,:,i]-MR[:,:,i]
        t1[t1<0]=-1#energy limited
        t1[t1>0]=1#water limited
        T="Water-Energy Zones For Year "+str(1951+i)
        plot_hm(t1,T)



def part2():
    """
    Expansion/Reduction
    """

    MPET,MR=yearly()
    E,W,Y=[],[],[]
    for i in range(64):
        t1=MPET[:,:,i]-MR[:,:,i]
        t1[t1<0]=-1#energy limited
        t1[t1>0]=1#water limited
        val,freq=np.unique(t1,return_counts=True)        
        E.append(freq[0]/(freq[0]+freq[2]))
        W.append(freq[2]/(freq[0]+freq[2]))
        Y.append(1951+i)

    ind=np.arange(64)
    width=1
    p1=plt.bar(ind,E,width)
    p2=plt.bar(ind,W,width,bottom=E)
    plt.ylabel('Percent')
    plt.title('Expansion/Reduction')
    plt.xticks(ind,Y,rotation=90)
    plt.yticks(np.arange(0,1.1,.1))
    plt.legend((p1[0],p2[0]),('Energy Limited','Water Limited'))
    plt.show()



def zone_anal(data):
    """
    Do zonal analysis of water-energy zones 
    """

    import seaborn as sns
    # plt.scatter(data['X'],data['Y'],s=1)
    ax=sns.heatmap(data['year'],cmap="YlGnBu")
    plt.show()



    pass



def part3():
    """
    Grid point change
    """

    M=np.zeros(121*121*64).reshape((121,121,64))
    MPET,MR=yearly()
    for i in range(64):
        t1=MPET[:,:,i]-MR[:,:,i]
        t1[t1<0]=-1#energy limited
        t1[t1>0]=1#water limited
        M[:,:,i]=t1

    dtype=[('X',np.uint8),('Y',np.uint8),('-1',np.uint8),('1',np.uint8),('year','64int8')]
    data=np.zeros(0,dtype=dtype)
    for i in range(121):
        for j in range(121):
            if np.unique(M[i][j]).size > 1:
                uniq,ind,c=np.unique(M[i][j],return_inverse=True,return_counts=True)
                data=np.append(data,np.array([(i,j,c[0],c[1],ind)],dtype=data.dtype))

    zone_anal(data)



if __name__ == "__main__":
    
    PET,R=load_data()
    # (121, 121, 768) 64 years monthly data

    # part1()
    # part2()
    part3()
