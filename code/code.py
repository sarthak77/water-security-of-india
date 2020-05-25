from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns


def load_data():
    """
    Return data in np format
    """

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
            if np.unique(M[i][j]).size > 1 or not np.all(M[i][j]):
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
    8 = complete mask file
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
    if zone == 8:
        return temp
    temp[temp==0]=zone
    temp=temp == zone 
    temp=np.invert(temp)
    return temp



def plot_hm(X,t,zone=7):
    """
    Plot heat map
    """

    global T
    X=X.transpose()
    X=np.flipud(X)
    # mask=get_mask()#for changing grid points
    mask=zone_mask(zone)#for individual zones
    ax=sns.heatmap(X,xticklabels=False,yticklabels=False,cmap="YlGnBu",cbar=False,mask=mask)
    # ax=sns.heatmap(X,xticklabels=False,yticklabels=False,cmap="YlGnBu",mask=mask)
    # ax=sns.heatmap(X,xticklabels=False,yticklabels=False,cmap="YlGnBu",cbar=False)
    
    plt.title(t+" for "+T[zone])
    # plt.show()
    # exit(-1)
    plt.savefig("images/zone/"+str(zone)+"/"+t+".png")
    plt.close()



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
    Yearly zone variation (64 plots each)
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
    Global Expansion/Reduction
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



def plot_india_zone(colors,T):
    """
    Plot zones of India
    """

    mask=zone_mask(8)
    mask=np.rot90(mask,k=3)
    C,X,Y=[[],[],[]]
    for i in range(121):
        for j in range(121):
            X.append(i)
            Y.append(j)
            C.append(colors[mask[i][j]])

    plt.scatter(X,Y,c=C)
    plt.title("Zones in India")
    handles=[]
    for i in range(7):
        handles.append(mpatches.Patch(color=colors[i+1],label=T[i+1]))
    plt.legend(handles=handles)
    plt.show()



def plot_indi_zones(colors,T,data):
    """
    Plot individual zones on map of India
    """

    for i in range(1,8,1):
        mask=zone_mask(i)
        mask=np.rot90(mask,k=3)

        C=[]
        for j,k in zip(data['X'],data['Y']):
            if not mask[j][k]:
                C.append(colors[i])
            else:
                C.append('black')

        plt.scatter(data['X'],data['Y'],s=8,c=C)
        plt.title(T[i])
        plt.savefig("images/zonal_analysis/"+str(i)+"/"+T[i]+".png")



def get_zone(data):
    """
    Return zone of each point
    """

    Z=[[] for i in range(7)]
    mask=zone_mask(8)
    mask=np.rot90(mask,k=3)
    for i in range(len(data)):
        x=data[i]['X']
        y=data[i]['Y']
        Z[mask[x][y]-1].append(i)
    return Z



def indi_zone_anal(colors,T,data):
    """
    Individual zone analysis
    """

    def year_hm(z,t):
        """
        Plot yearly grid point variation
        """

        plt.rcParams.update({'font.size': 22})
        ax=sns.heatmap(z['year'],xticklabels=list(range(1951,2015,1)),cmap="YlGnBu",cbar=False)
        plt.title(T[i+1]+" Variation with year",fontsize=30)
        ax.set_xlabel("Year",fontsize=25)
        ax.set_ylabel("Number of Points",fontsize=25)
        figure = plt.gcf()
        figure.set_size_inches(32, 18)
        plt.savefig("images/zonal_analysis/"+t+T[i+1]+" Yearly grid point variation"+".png")
        plt.close()

    def year_sbc(z,t):
        """
        Plot yearly zone variation stacked bar chart
        """

        wl=np.sum(z['year'],axis=0)
        el=np.subtract(np.full((64),z.shape[0]),wl)
        ind=np.arange(64)
        width=1
        
        plt.rcParams.update({'font.size': 22})
        p1=plt.bar(ind,el,width)
        p2=plt.bar(ind,wl,width,bottom=el)
        plt.ylabel('Number of Points',fontsize=25)
        plt.xlabel('Year',fontsize=25)
        plt.title('Water-Energy Zones for '+T[i+1],fontsize=30)
        plt.xticks(ind,list(range(1951,2015,1)),rotation=90)
        plt.legend((p1[0],p2[0]),('Energy Limited','Water Limited'))
        figure = plt.gcf()
        figure.set_size_inches(32, 18)
        plt.savefig("images/zonal_analysis/"+t+T[i+1]+" Yearly zone variation"+".png")
        plt.close()



    Z=get_zone(data)

    for i in range(7):
        z=np.array([data[j] for j in Z[i]])
        year_hm(z,str(i+1)+"/")
        year_sbc(z,str(i+1)+"/")





    pass



def zone_analysis(data):
    """
    Do zonal analysis of water-energy zones 
    0 = sea zone
    1 = central zone
    2 = north-east zone
    3 = north-east-hills zone
    4 = south zone
    5 = J&K zone
    6 = west  zone
    7 = north zone
    """

    global T
    colors = cm.rainbow(np.linspace(0,1,8)) 

    # plot_india_zone(colors,T)
    # plot_indi_zones(colors,T,data)
    indi_zone_anal(colors,T,data)



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

    # zone_analysis(data)
    zyv(data)
    # gpv(data)



def gpv(data):
    """
    plot heat map of changing grid points
    """

    global T
    t=np.zeros(121*121).reshape((121,121))
    for i in data:
        for j in range(len(i['year'])):
            if j:
                if i['year'][j] != i['year'][j-1]:
                    t[i['X']][i['Y']]+=1

    # plot_hm(t,"Gird point varaition in India")
    for i in range(1,8,1):
        plot_hm(t,"Gird point varaition in "+T[i],i)



def zyv(data):
    """
    Visualize global/zonal change in grid points vs year
    """

    global T
    D=[data]
    Z=get_zone(data)

    for i in range(7):
        D.append(np.array([data[j] for j in Z[i]]))

    for ind in range(len(D)):
    
        X=list(range(1952,2015,1))
        Y=np.zeros(63)
        d=D[ind]

        for i in d:
            for j in range(len(i['year'])):
                if j:
                    if i['year'][j] != i['year'][j-1]:
                        Y[j-1]+=1

        plt.rcParams.update({'font.size': 22})
        plt.bar(X,Y,tick_label=X)
        plt.xlabel("Year",fontsize=25)
        plt.ylabel("Number of Points",fontsize=25)
        plt.xticks(rotation=90)
        figure = plt.gcf()
        figure.set_size_inches(32, 18)

        # if ind == 0:
            # plt.title("Yearly change of grid point in India",fontsize=30)
            # plt.savefig("images/india/"+"Yearly change of grid points in "+"India"+".png")
        if ind:
            plt.title("Yearly change of grid point in "+T[ind],fontsize=30)
            plt.savefig("images/zonal_analysis/"+str(ind)+"/Yearly change of grid points in "+T[ind]+".png")
        plt.close()



if __name__ == "__main__":
    
    PET,R=load_data()
    # (121, 121, 768) 64 years monthly data

    T={1:"Central Zone",2:"North East Zone",3:"North Eastern Hills Zone",4:"South Zone",5:"J&K Zone",6:"West Zone",7:"North Zone"}
    part1()
    # part2()
    # part3()
