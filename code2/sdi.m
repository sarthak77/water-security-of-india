clc;
%%%% Following code is the implementation of SDI for two 3-D matrices where
%%%% the 1st and 2nd dimensions of the matrices are the x-coordinate and y-coordinate respectively
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
for i = 1:121 %x-coordinate
   for j = 1:121 %y-coordinate
        for k = 1:64 %year

     pcp_pet = squeeze(MPET(i,j,k)) - squeeze(MR(i,j,k)); %precipitation - eto for every grid node, month, and year
     pet_pcp = squeeze(MR(i,j,k))- squeeze(MPET(i,j,k)); %eto- precipitation for every grid node, month, and year

     SurINT_step1 =  pcp_pet; %define the first step to calculate SurINT
     SurINT_step1 (SurINT_step1 < 0) = NaN;  %remove all negative (or deficit values)
     SurINT(i,j,k) = nanmean(SurINT_step1); %calculate SurINT for each year
    
     DT_step1 = pet_pcp; %define the first step to calculate DT
     DT_step1(DT_step1> 0) = 0; % set the deficit values to zero
     f = find(diff([1,transpose(squeeze(DT_step1)),1]==0));%first the starting point and end point of each deficit time period within an indvidual year
     p = f(1:2:end-1);  % first the starting point each deficit time period within an individual year
     DT(i,j,k)= nanmean(f(2:2:end)-p); %takes the mean of the deficit time duration for each event for an individual year
        end
    end
end

SurINT_z= zscore(SurINT,[],3); %calculates z-score of SurINT for the entire time period
DT_z = zscore(DT,[],3);  %calculates z-score of DT for the entire time period

SDI_step1 = (DT_z+ SurINT_z); %top of Equation 1 
SDI = SDI_step1./std((DT_z+ SurINT_z),[],3);  %calculates SDI for the entire time period