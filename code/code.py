from scipy.io import loadmat
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
import copy


def load_data():
    """
    Return data in np format
    """

    f1='water-security-of-india/code/Datasets/PET_India.mat'
    f2='water-security-of-india/code/Datasets/rain_India.mat'
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
            # if np.unique(M[i][j]).size > 1 or not np.all(M[i][j]):
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
    8 = complete mask file
    """

    with open('water-security-of-india/code/mask/homogeneouszones.txt', 'r') as file:
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



def plot_hm(X,t,zone=0):
    """
    Plot heat map
    """

    global T
    X=X.transpose()
    X=np.flipud(X)
    # mask=get_mask()#for changing grid points
    mask=zone_mask(zone)#for individual zones
    # ax=sns.heatmap(X,xticklabels=False,yticklabels=False,cmap="YlGnBu",cbar=False,mask=mask)
    ax=sns.heatmap(X,xticklabels=False,yticklabels=False,cmap="YlGnBu",vmin=0,vmax=10)
    # ax=sns.heatmap(X,xticklabels=False,yticklabels=False,cmap="YlGnBu",mask=mask)
    # ax=sns.heatmap(X,xticklabels=False,yticklabels=False,cmap="YlGnBu",cbar=False)
    
    plt.title(t)
    plt.show()
    exit(-1)
    # plt.savefig("images/temp/"+t+".png")
    # plt.savefig("images/zonal_analysis/"+str(zone)+"/"+t+".png")
    # plt.close()



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
    plt.ylabel('Percent of Grid Points')
    plt.title('Water-Energy Zones for India')
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
        plt.savefig("water-security-of-india/code/images/zonal_analysis/"+str(i)+"/"+T[i]+".png")



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
        plt.title(T[i+1]+": Individual grid point variation with year",fontsize=30)
        ax.set_xlabel("Year",fontsize=25)
        ax.set_ylabel("Number of Points",fontsize=25)
        figure = plt.gcf()
        figure.set_size_inches(32, 18)
        # plt.show()
        plt.savefig("water-security-of-india/code/images/zonal_analysis/"+t+T[i+1]+" Yearly grid point variation"+".png")
        plt.close()

    def year_sbc(z,t):
        """
        Plot yearly zone variation stacked bar chart
        """

        wl=np.sum(z['year'],axis=0)
        wl=np.true_divide(wl,z.shape[0])
        el=np.subtract(np.full((64),1),wl)

        ind=np.arange(64)
        width=1
        
        plt.rcParams.update({'font.size': 22})
        p1=plt.bar(ind,el,width)
        p2=plt.bar(ind,wl,width,bottom=el)
        plt.ylabel('Percent of Grid Points',fontsize=25)
        plt.xlabel('Year',fontsize=25)
        plt.title('Water-Energy Zones for '+T[i+1],fontsize=30)
        plt.xticks(ind,list(range(1951,2015,1)),rotation=90)
        plt.legend((p1[0],p2[0]),('Energy Limited','Water Limited'))
        figure = plt.gcf()
        figure.set_size_inches(32, 18)
        # plt.show()
        plt.savefig("water-security-of-india/code/images/zonal_analysis/"+t+T[i+1]+" Yearly zone variation"+".png")
        plt.close()



    Z=get_zone(data)

    for i in range(7):
        z=np.array([data[j] for j in Z[i]])
        year_hm(z,str(i+1)+"/")
        year_sbc(z,str(i+1)+"/")



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
    # zyv(data)
    # gpv(data)
    return data



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

    # plot_hm(t,"Frequency of gird point varaition in India")
    for i in range(1,8,1):
        plot_hm(t,"Frequency of gird point varaition in "+T[i],i)



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

        Y=np.true_divide(Y,d.shape[0])
        plt.rcParams.update({'font.size': 22})
        plt.bar(X,Y,tick_label=X)
        plt.xlabel("Year",fontsize=25)
        plt.ylabel("Percent of total Points",fontsize=25)
        plt.xticks(rotation=90)
        figure = plt.gcf()
        figure.set_size_inches(32, 18)

        if ind == 0:
            plt.title("Yearly change of grid point in India",fontsize=30)
            plt.savefig("images/india/"+"Yearly change of grid points in "+"India"+".png")
        if ind:
            plt.title("Yearly change of grid point in "+T[ind],fontsize=30)
            plt.savefig("images/zonal_analysis/"+str(ind)+"/Yearly change of grid points in "+T[ind]+".png")
        plt.close()



def plots(P,PET,R,T,zone=0):
    """
    sdsdsds
    """

    args=[]
    for i in [P,PET,R]:
        t=np.transpose(i)
        t=np.flipud(t)
        args.append(t)

    plt.rcParams.update({'font.size': 22})
    grid_kws = {"height_ratios": (.9, .03), "hspace": .3}
    fig,axes=plt.subplots(2,3,gridspec_kw=grid_kws)
    fig.suptitle(T)
    # M=np.invert(np.array(args[0],dtype=bool))
    M=zone_mask(zone)

    sns.heatmap(args[0],xticklabels=False,yticklabels=False,cmap="YlGnBu",ax=axes[0,0],cbar_ax=axes[1,0],cbar_kws={"orientation":"horizontal"},mask=M,vmin=0,vmax=200)
    sns.heatmap(args[1],xticklabels=False,yticklabels=False,cmap="YlGnBu",ax=axes[0,1],cbar_ax=axes[1,1],cbar_kws={"orientation":"horizontal"},mask=M,vmin=120,vmax=150)
    sns.heatmap(args[2],xticklabels=False,yticklabels=False,cmap="YlGnBu",ax=axes[0,2],cbar_ax=axes[1,2],cbar_kws={"orientation":"horizontal"},mask=M,vmin=0,vmax=6)

    axes[0,0].set_title("P")
    axes[0,1].set_title("PET")
    axes[0,2].set_title("PET/P")

    figure = plt.gcf()
    figure.set_size_inches(32, 18)

    # plt.show()
    # exit(-1)
    plt.savefig("images/aridity/"+str(zone)+"/"+T+".png")



def PT(X,Y,b1):
    """
    Return change point for array X using pettitt test
    """

    n=X.shape[0]
    CP=[]
    for i in range(n):
        b2=np.array([j%64<i for j in range(64*64)]).reshape((64,64))
        b3=np.multiply(b1,b2)
        ret=0
        t1=X[:,None]>X
        t2=np.multiply(b3,t1)
        ret+=np.sum(t2)
        t1=X[:,None]<X
        t2=np.multiply(b3,t1)
        ret-=np.sum(t2)
        CP.append(ret)
    
    T=np.argmax(CP)
    K=CP[T]
    sl=0.5#define significance level
    t=64
    p=2*np.exp(-6*K**2/(t**3+t**2))

    if(p<sl):
        return T+Y
    else:
        return 0



def Petit(PET,P):
    """
    Calculate change point using petit test
    """

    # For PET
    d1={0:0}
    x=np.array([i>j for i in range(64) for j in range(64)]).reshape((64,64))
    for i in range(121):
        for j in range(121):
            t=PT(PET[i][j],1951,x)
            if t in d1.keys():
                d1[t]+=1
            else:
                d1[t]=1

    # For P
    d2={0:0}
    x=np.array([i>j for i in range(64) for j in range(64)]).reshape((64,64))
    for i in range(121):
        for j in range(121):
            t=PT(P[i][j],1951,x)
            if t in d2.keys():
                d2[t]+=1
            else:
                d2[t]=1

    print(d1)
    print(d2)



def Petitz(zone,PET,P):
    """
    Petit test for zonal data
    """

    data=part3()
    Z=get_zone(data)
    z=np.array([data[j] for j in Z[zone-1]])

    # For PET
    d1={0:0}
    x=np.array([i>j for i in range(64) for j in range(64)]).reshape((64,64))
    for i,j in zip(z['X'],z['Y']):
        t=PT(PET[i][j],1951,x)
        if t in d1.keys():
            d1[t]+=1
        else:
            d1[t]=1

    # For P
    d2={0:0}
    x=np.array([i>j for i in range(64) for j in range(64)]).reshape((64,64))
    for i,j in zip(z['X'],z['Y']):
        t=PT(P[i][j],1951,x)
        if t in d2.keys():
            d2[t]+=1
        else:
            d2[t]=1

    print(d1)
    print(d2)



def part4():
    """
    Aridity Analysis
    """

    MPET,MR=yearly()
    # y=Petit(MPET,MR)
    # y=Petitz(7,MPET,MR)#for zonal data
    # y=1964 for India
    CP=[2009,2013,1969,2011,1986,1985,1964]
    
    for i in range(len(CP)):
        y=CP[i]-1951

        t1=np.sum(MR[:,:,0:y],axis=2)
        t1=np.true_divide(t1,12*y)

        t2=np.sum(MPET[:,:,0:y],axis=2)
        t2=np.true_divide(t2,12*y)

        t=copy.deepcopy(t1)
        t[t==0]=1
        t=np.divide(t2,t)
        plots(t1,t2,t,"Before "+str(1951+y),i+1)


        t1=np.sum(MR[:,:,y:],axis=2)
        t1=np.true_divide(t1,12*(64-y))

        t2=np.sum(MPET[:,:,y:],axis=2)
        t2=np.true_divide(t2,12*(64-y))

        t=copy.deepcopy(t1)
        t[t==0]=1
        t=np.divide(t2,t)
        plots(t1,t2,t,"After "+str(1951+y),i+1)



if __name__ == "__main__":
    
    PET,R=load_data()
    # (121, 121, 768) 64 years monthly data

    T={1:"Central Zone",2:"North East Zone",3:"North Eastern Hills Zone",4:"South Zone",5:"J&K Zone",6:"West Zone",7:"North Zone"}
    # part1()
    # part2()
    # part3()
    part4()
