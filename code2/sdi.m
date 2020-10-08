clc;
%%%% Following code is the implementation of SDI for two 3-D matrices where
%%%% the 1st and 2nd dimensions of the matrices are the coordinates of a point
%%%% and the 3rd dimension is monthly data

%% Loading the data %%
R = load("rain_India.mat");
PET = load("PET_India.mat");
R = R.rain;
PET = PET.PET;

%% Converting monthly data to yearly data %%
% converting 768 months into 64 years and storing the data in Monthly PET
% (MPET) and Monthly R(MR) matrices.

MPET = zeros(121,121,64);   %declare matrix of size 121x121x64
MR = zeros(121,121,64); %declare matrix of size 121x121x64
for i=1:64  %year
    t1 = PET(:,:,12*(i-1)+1:12*i);  %taking data of 12 months at a time for PET
    t = sum(t1,3);  %combining the data for entire time period
    MPET(:,:,i) = t;    %storing the data of ith year in MPET
    
    t1 = R(:,:,12*(i-1)+1:12*i);    %taking data of 12 months at a time for R
    t = sum(t1,3);  %combining the data for entire time period
    MR(:,:,i) = t;  %storing the data of ith year in MR
end

%% Calculating Surplus and Deficiency %%
SurINT = zeros(121,121);    %declare matrix of size 121x121 for storing Surplus intensity
DT = zeros(121,121);    %declare matrix of size 121x121 for storing Deficit time

for i=1:121 %x coordinate
    for j=1:121 %y coordinate
        SurINT_st1 = MR(i,j,:) - MPET(i,j,:);   %First step to calculate SurINT, precipitation-et0
        SurINT_st1(SurINT_st1<0) = NaN; %remove all negative or deficit values
        SurINT(i,j) = nanmean(SurINT_st1);  %SurINT for each coordinate over all the years
        
        DT_st1 = MPET(i,j,:) - MR(i,j,:);   %First step to calculate DT, et0-precipitation
        DT_st1(DT_st1<0) = NaN;  %remove all negative or surplus values
        DT(i,j) = nanmean(DT_st1);  %DT for each coordinate over all the years
        
        
    end
end

SurINT_z = zscore(SurINT); %z-score of SurINT
DT_z = zscore(DT); %z-score of DT

SDI_st1 = (DT_z + SurINT_z);    %top part of SDI equation
SDI = SDI_st1/std((DT_z+SurINT_z)); %Final SDI