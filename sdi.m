clc;
%% Loading the data %%
R = load("rain_India.mat");
PET = load("PET_India.mat");
R = R.rain;
PET = PET.PET;

%% Converting monthly data to yearly data %%
MPET = zeros(121,121,64);
MR = zeros(121,121,64);
for i=1:64
    t1 = PET(:,:,12*(i-1)+1:12*i);
    t = sum(t1,3);
    MPET(:,:,i) = t;
    
    t1 = R(:,:,12*(i-1)+1:12*i);
    t = sum(t1,3);
    MR(:,:,i) = t;
end

%% Calculating Surplus and Deficiency %%
M = zeros(121,121);
SurINT = zeros(121,121);
DT = zeros(121,121);

for i=1:121
    for j=1:121
        SurINT_st1 = MR(i,j,:) - MPET(i,j,:);
        SurINT_st1(SurINT_st1<0) = 0;
        SurINT(i,j) = nanmean(SurINT_st1);
        
        DT_st1 = MPET(i,j,:) - MR(i,j,:);
        DT_st1(DT_st1<0) = 0;
        DT(i,j) = nanmean(DT_st1);
     
    end
    %SurINT_z = zscore(SurINT,[],2);
    %DT_z = zscore(DT,[],2);
        
    %SDI_st1 = (DT_z + SurINT_z);
    %M(i,:) = SDI_st1/std(DT_z+SurINT_z);
end

SurINT_z = zscore(SurINT);
DT_z = zscore(DT);

SDI_st1 = (DT_z + SurINT_z);
SDI = SDI_st1/std((DT_z+SurINT_z));
