%%%This script generates the components of Figure 7 Panel A, from the publication: 
% Lafer-Sousa, R., & Conway, B. R. (2017). #thedress: Categorical perception of an ambiguous color image. Journal of Vision
% Figure legend:
% Figure 7.  Categorical verbal reports can be predicted from color matches.  
% Multinomial logistic regression was used to build nominal response models 
% (classifiers). Four models were generated: the ?full? model was fit using 
% the L*, u*, and v* components of subjects? color matches (to all 4 match regions)
% as predictors; three additional models were fit using either the L* or u* or v* 
% component of subjects? matches as predictors.  Models were fit with responses 
% from a subset of the online subjects (half the subjects from Experiment 2,
% Ntrain = 549) and tested on responses from the left-out subjects (Ntest = 547). 
% A, Predicted probability of category membership for the ?full? model.  
% Each panel contains the results for data from individual (left-out) subjects, 
% grouped by the verbal label they used to describe the dress (ground truth), 
% and whether they had seen the dress prior to the study (?Naïve?).  Each thin
% vertical column within a panel shows the results for a single subject: 
% the colors of each row in the column represent the predicted probability that 
% the subject used each of the categorical labels (?B/K?, ?W/G?,?B/B?,?Other?); 
% each column sums to 1.  Subjects are rank-ordered by the predicted probability 
% for the ground-truth class. The average predicted probabilities for each 
% response category are denoted by (?P).  B, Bar plots quantifying classification 
% performance (the area under the receiver operating characteristic curves, 
% computed using the true positive rates and false positive rates), by category, 
% for each of the 4 models.  Error bars indicate 95% C.I. Values greater than 
% 0.90 indicate ?excellent? performance; values between 0.75-0.90 ? ?fair to 
% good? performance; values below 0.5-0.75 ? ?poor? performance. We compared 
% the accuracy of the various models against each other using Matlab?s 
% testcholdout function:  the ?L* only? model performed no better than the 
% ?u* only? model (Naïve: p = 0.5; Not-Naïve: p = 0.8), or the ?v* only? model 
% (Naïve: p = 0.5; Not-Naïve: p = 0.3). The ?full? model was more accurate than 
% the  ?L* only? model, but only among ?Naïve? subjects (Naïve: p<0.001; 
% Not-Naive: p = 0.09).  True positive rates (Sensitivity) for all four models 
% are provided in Table 1.


%%%The script was written by Rosa Lafer-Sousa. (for questions, contact: rlaferso@mit.edu)
    % Note: this code was modified from its original format to produce only the contents of Figure 1 from the publication. last
    % modified: 10/13/2017.  Matlab version 2016B 

%%%Notes on running the code:
% The code loads data from several .mat files that were generated from the raw data provided in the folder RAW DATA:
    % data collected online (Amazon's Mechanical Turk) from the 'SCALE Experiment' 
        %  TURK_SCALE_10_36_100_150_BK.mat   'Blue/Black' reporters
        %  TURK_SCALE_10_36_100_150_WG.mat   'White/Gold' reporters
        %  TURK_SCALE_10_36_100_150_BG.mat   'Blue/Brown' reporters
        %  TURK_SCALE_10_36_100_150_OT.mat   'Other color' reporters
   
% Running this script will generate 1 figure file, containing panel A of the published figure.
 

clear all; close all; clc

%Choose model predictors:
use_LUV_UV_U_V_or_L = 0; %0 = LUV; 1-UV, 2-U; 3-V; L-4 . choose which components of matches to use as predictors to train  the model 
use_upper_lower_both = 3; %1=upper, 2=lower, 3=both; choose match regions


binarize=0; %set to 1 to binarize classification, 0 to use gradient classification (sum(pihat))
if use_upper_lower_both ==1
    r_inds = 1:2;
    LUV_inds =  [1 2 5 6 9 10];
elseif use_upper_lower_both ==2
    r_inds = 3:4;
    LUV_inds =  [3 4 7 8 11 12];
elseif use_upper_lower_both ==3
    r_inds = 1:4;
     LUV_inds =  1:12;
end
interactions = 1; % 1 for interactions - always use  for multinomial regression

%%TRAIN and TEST the MODEL:
filenames =   {'TURK_SCALE_10_36_100_150_BK';...
        'TURK_SCALE_10_36_100_150_WG';...
        'TURK_SCALE_10_36_100_150_BG';...
        'TURK_SCALE_10_36_100_150_OT'};%all subjects from the SCALE expt only (all scales presented) - using these subjects because they did not experience the disambiguating stimuli prior to match-making

load(filenames{1}) ;

LBK = Luv_L_by_region(:,r_inds);
UBK = Luv_u_by_region(:,r_inds);
VBK = Luv_v_by_region(:,r_inds);
LUV_BK = [LBK UBK VBK];

load(filenames{2}) ;
LWG = Luv_L_by_region(:,r_inds);
UWG = Luv_u_by_region(:,r_inds);
VWG = Luv_v_by_region(:,r_inds);
LUV_WG = [LWG UWG VWG];

load(filenames{3}) ;
LBG = Luv_L_by_region(:,r_inds);
UBG = Luv_u_by_region(:,r_inds);
VBG = Luv_v_by_region(:,r_inds);
LUV_BG = [LBG UBG VBG];

load(filenames{4}) ;
LOT = Luv_L_by_region(:,r_inds);
UOT = Luv_u_by_region(:,r_inds);
VOT = Luv_v_by_region(:,r_inds);
LUV_OT = [LOT UOT VOT];

numBK = size(LUV_BK,1);
numWG = size(LUV_WG,1);
numBG = size(LUV_BG,1);
numOT = size(LUV_OT,1);

%%select training samples
LUV = [LUV_BK; LUV_WG; LUV_BG; LUV_OT];% 
UV = [UBK VBK; UWG VWG; UBG VBG; UOT VOT];% 
U = [UBK; UWG; UBG; UOT];
V = [VBK; VWG; VBG; VOT];
L = [LBK; LWG; LBG; LOT];

if  use_LUV_UV_U_V_or_L == 0
        LUV = LUV;
elseif use_LUV_UV_U_V_or_L == 1
        LUV= UV;
elseif use_LUV_UV_U_V_or_L == 2
        LUV = U;
elseif use_LUV_UV_U_V_or_L == 3
        LUV = V;
elseif use_LUV_UV_U_V_or_L == 4
        LUV = L;
end

 
CATS = [ones(size(LUV_BK,1),1); ones(size(LUV_WG,1),1)*2;...
    ones(size(LUV_BG,1),1)*3; ones(size(LUV_OT,1),1)*4];%
NAIVE = [indices_naive(cat_resps==1); indices_naive(cat_resps==2);...
    indices_naive(cat_resps==3); indices_naive(cat_resps==4)];
NOTNAIVE = [indices_exp(cat_resps==1); indices_exp(cat_resps==2);...
    indices_exp(cat_resps==3); indices_exp(cat_resps==4)];

sc=NUM(:,39); %scale 
sc_36 = sc==36;
SCALE_36 = [sc_36(cat_resps==1); sc_36(cat_resps==2);...
    sc_36(cat_resps==3); sc_36(cat_resps==4)];
sc_100 = sc==100;
SCALE_100 = [sc_100(cat_resps==1); sc_100(cat_resps==2);...
    sc_100(cat_resps==3); sc_100(cat_resps==4)];
sc_10_100 = sc==10|sc==100;
SCALE_10_100 = [sc_10_100(cat_resps==1);sc_10_100(cat_resps==2);...
  sc_10_100(cat_resps==3);sc_10_100(cat_resps==4)];
sc_10_150 = sc==10|sc==150;
SCALE_10_150 = [sc_10_150(cat_resps==1);sc_10_150(cat_resps==2);...
  sc_10_150(cat_resps==3);sc_10_150(cat_resps==4)];
sc_36_100 = sc==36|sc==100;
SCALE_36_100 = [sc_36_100(cat_resps==1);sc_36_100(cat_resps==2);...
  sc_36_100(cat_resps==3);sc_36_100(cat_resps==4)];
sc_36_150 = sc==36|sc==150;
SCALE_36_150 = [sc_36_150(cat_resps==1);sc_36_150(cat_resps==2);...
  sc_36_150(cat_resps==3);sc_36_150(cat_resps==4)];
sc_10_36_150 = sc==10|sc==36|sc==150;
SCALE_10_36_150 = [sc_10_36_150(cat_resps==1);sc_10_36_150(cat_resps==2);...
  sc_10_36_150(cat_resps==3);sc_10_36_150(cat_resps==4)];

%SELECT TESTING AND TRAINING SAMPLES
%if test_with_scale_36 == 3
testing_sample_inds = SCALE_10_100;%taking data from scale 10% and 100% conditions to train, and testing with samples from scale 36% and 150% conditions (these are independent subjects and data, and yield ~50/50 split of the scale experiment data set)
training_sample_inds = 1:length(CATS);
training_sample_inds(testing_sample_inds) = []; %remove testing samples

LUV_TRS = LUV(training_sample_inds,:); %training samples
CATS_TRS = CATS(training_sample_inds); %training sample group membership (bk, wg, bg, ot)
NAIVE_TRS = NAIVE(training_sample_inds); %training sample naive indices
NOTNAIVE_TRS = NOTNAIVE(training_sample_inds); %training sample not-naive indices
LUV_TES = LUV(testing_sample_inds,:); %testing samples
CATS_TES = CATS(testing_sample_inds); %testing sample group membership (bk, wg, bg, ot)
NAIVE_TES = NAIVE(testing_sample_inds); %training sample naive indices
NOTNAIVE_TES = NOTNAIVE(testing_sample_inds); %training sample not-naive indices
%end

NAIVE_CATS_TRS = CATS_TRS(NAIVE_TRS);
NOTNAIVE_CATS_TRS = CATS_TRS(NOTNAIVE_TRS);
NAIVE_LUV_TRS = LUV_TRS(NAIVE_TRS,:);
NOTNAIVE_LUV_TRS = LUV_TRS(NOTNAIVE_TRS,:);
NAIVE_CATS_TES = CATS_TES(NAIVE_TES);
NOTNAIVE_CATS_TES = CATS_TES(NOTNAIVE_TES);
NAIVE_LUV_TES = LUV_TES(NAIVE_TES,:);
NOTNAIVE_LUV_TES = LUV_TES(NOTNAIVE_TES,:);

LUV_BK_TRS = LUV_TRS(CATS_TRS==1,:);
LUV_WG_TRS = LUV_TRS(CATS_TRS==2,:);
numBK_TRS = sum(CATS_TRS==1);
numWG_TRS = sum(CATS_TRS==2);

LUV_BG_TRS = LUV_TRS(CATS_TRS==3,:);
LUV_OT_TRS = LUV_TRS(CATS_TRS==4,:);
numBG_TRS = sum(CATS_TRS==3);
numOT_TRS = sum(CATS_TRS==4);

LUV_BK_TES = LUV_TES(CATS_TES==1,:);
LUV_WG_TES = LUV_TES(CATS_TES==2,:);
numBK_TES = sum(CATS_TES==1);
numWG_TES = sum(CATS_TES==2);

LUV_BG_TES = LUV_TES(CATS_TES==3,:);
LUV_OT_TES = LUV_TES(CATS_TES==4,:);
numBG_TES = sum(CATS_TES==3);
numOT_TES = sum(CATS_TES==4);


NAIVE_LUV_BK_TRS = NAIVE_LUV_TRS(NAIVE_CATS_TRS==1,:);
NAIVE_LUV_WG_TRS = NAIVE_LUV_TRS(NAIVE_CATS_TRS==2,:);
NAIVE_numBK_TRS = sum(NAIVE_CATS_TRS==1);
NAIVE_numWG_TRS = sum(NAIVE_CATS_TRS==2);

NAIVE_LUV_BG_TRS = NAIVE_LUV_TRS(NAIVE_CATS_TRS==3,:);
NAIVE_LUV_OT_TRS = NAIVE_LUV_TRS(NAIVE_CATS_TRS==4,:);
NAIVE_numBG_TRS = sum(NAIVE_CATS_TRS==3);
NAIVE_numOT_TRS = sum(NAIVE_CATS_TRS==4);

NAIVE_LUV_BK_TES = NAIVE_LUV_TES(NAIVE_CATS_TES==1,:);
NAIVE_LUV_WG_TES = NAIVE_LUV_TES(NAIVE_CATS_TES==2,:);
NAIVE_numBK_TES = sum(NAIVE_CATS_TES==1);
NAIVE_numWG_TES = sum(NAIVE_CATS_TES==2);

NAIVE_LUV_BG_TES = NAIVE_LUV_TES(NAIVE_CATS_TES==3,:);
NAIVE_LUV_OT_TES = NAIVE_LUV_TES(NAIVE_CATS_TES==4,:);
NAIVE_numBG_TES = sum(NAIVE_CATS_TES==3);
NAIVE_numOT_TES = sum(NAIVE_CATS_TES==4);


NOTNAIVE_LUV_BK_TRS = NOTNAIVE_LUV_TRS(NOTNAIVE_CATS_TRS==1,:);
NOTNAIVE_LUV_WG_TRS = NOTNAIVE_LUV_TRS(NOTNAIVE_CATS_TRS==2,:);
NOTNAIVE_numBK_TRS = sum(NOTNAIVE_CATS_TRS==1);
NOTNAIVE_numWG_TRS = sum(NOTNAIVE_CATS_TRS==2);

NOTNAIVE_LUV_BG_TRS = NOTNAIVE_LUV_TRS(NOTNAIVE_CATS_TRS==3,:);
NOTNAIVE_LUV_OT_TRS = NOTNAIVE_LUV_TRS(NOTNAIVE_CATS_TRS==4,:);
NOTNAIVE_numBG_TRS = sum(NOTNAIVE_CATS_TRS==3);
NOTNAIVE_numOT_TRS = sum(NOTNAIVE_CATS_TRS==4);

NOTNAIVE_LUV_BK_TES = NOTNAIVE_LUV_TES(NOTNAIVE_CATS_TES==1,:);
NOTNAIVE_LUV_WG_TES = NOTNAIVE_LUV_TES(NOTNAIVE_CATS_TES==2,:);
NOTNAIVE_numBK_TES = sum(NOTNAIVE_CATS_TES==1);
NOTNAIVE_numWG_TES = sum(NOTNAIVE_CATS_TES==2);

NOTNAIVE_LUV_BG_TES = NOTNAIVE_LUV_TES(NOTNAIVE_CATS_TES==3,:);
NOTNAIVE_LUV_OT_TES = NOTNAIVE_LUV_TES(NOTNAIVE_CATS_TES==4,:);
NOTNAIVE_numBG_TES = sum(NOTNAIVE_CATS_TES==3);
NOTNAIVE_numOT_TES = sum(NOTNAIVE_CATS_TES==4);


LUV_TRS;%this is the set of predictors
TERMS_TRS=categorical(CATS_TRS); %CATEGORICAL DEPENDENT VARIABLE (this is multinomial now... thing youre trying to predict: -- BK or WG or BG or OT)

if interactions == 1  %%DEFAULT for nominal and heirarchical mnrfits is interactions ON

      [B,dev,stats] = mnrfit(LUV_TRS,TERMS_TRS,'interactions','on');
     model_stem = {'_Nominal'};
       [pihat,dlow,dhi] = mnrval(B,LUV_TRS,stats);

elseif interactions == 0
    int_stem={'_interactionsOFF'};
         model_stem = {'_Nominal'};
      [B,dev,stats] = mnrfit(LUV_TRS,TERMS_TRS,'interactions','off');
  [pihat,dlow,dhi] = mnrval(B,LUV_TRS,stats,'interactions','off');
end



test_left_out = 0; %this variable will change as the code runs.
for t = 1:2
if test_left_out == 1    
    [pihat,dlow,dhi] = mnrval(B,LUV_TES,stats);%,'model','hierarchical');
    [pihat_NAIVE,dlow_NAIVE,dhi_NAIVE] = mnrval(B,NAIVE_LUV_TES,stats);
    [pihat_NOTNAIVE,dlow_NOTNAIVE,dhi_NOTNAIVE] = mnrval(B,NOTNAIVE_LUV_TES,stats);
    %find the distribution of bk, wg, bb other predicted in the left out data:
pihat_binarized_TES=[];
pihat_NAIVE_binarized_TES=[];
pihat_NOTNAIVE_binarized_TES=[];
for pih = 1:size(pihat,1)
    pihat_binarized_TES = [pihat_binarized_TES; pihat(pih,:) == max(pihat(pih,:))];
end
for pih = 1:size(pihat_NAIVE,1)
    pihat_NAIVE_binarized_TES = [pihat_NAIVE_binarized_TES; pihat_NAIVE(pih,:) == max(pihat_NAIVE(pih,:))];
end
for pih = 1:size(pihat_NOTNAIVE,1)
    pihat_NOTNAIVE_binarized_TES = [pihat_NOTNAIVE_binarized_TES; pihat_NOTNAIVE(pih,:) == max(pihat_NOTNAIVE(pih,:))];
end
if binarize==1
PIHAT_DIST_TES = sum(pihat_binarized_TES);
VERBAL_DIST_TES = [numBK_TES numWG_TES numBG_TES numOT_TES];
PIHAT_NAIVE_DIST_TES = sum(pihat_NAIVE_binarized_TES);
VERBAL_NAIVE_DIST_TES = [NAIVE_numBK_TES NAIVE_numWG_TES NAIVE_numBG_TES NAIVE_numOT_TES];
PIHAT_NOTNAIVE_DIST_TES = sum(pihat_NOTNAIVE_binarized_TES);
VERBAL_NOTNAIVE_DIST_TES = [NOTNAIVE_numBK_TES NOTNAIVE_numWG_TES NOTNAIVE_numBG_TES NOTNAIVE_numOT_TES];
else    
PIHAT_DIST_TES = sum(pihat);
VERBAL_DIST_TES = [numBK_TES numWG_TES numBG_TES numOT_TES];
PIHAT_NAIVE_DIST_TES = sum(pihat_NAIVE);
VERBAL_NAIVE_DIST_TES = [NAIVE_numBK_TES NAIVE_numWG_TES NAIVE_numBG_TES NAIVE_numOT_TES];
PIHAT_NOTNAIVE_DIST_TES = sum(pihat_NOTNAIVE);
VERBAL_NOTNAIVE_DIST_TES = [NOTNAIVE_numBK_TES NOTNAIVE_numWG_TES NOTNAIVE_numBG_TES NOTNAIVE_numOT_TES];
DHI_TES = sum(dhi)./sum(sum(pihat))*100;
DHI_NAIVE_TES = sum(dhi_NAIVE)./sum(sum(pihat_NAIVE))*100;
DHI_NOTNAIVE_TES = sum(dhi_NOTNAIVE)./sum(sum(pihat_NOTNAIVE))*100;
end       

output=[CATS_TES NOTNAIVE_TES pihat]; output_sorted = sortrows(output,[1 2]);
pihat_sorted2 = output_sorted;%(:,3:6); %this
CATS_TES_sorted2=output_sorted(:,1);
CATS_sorted2=CATS_TES_sorted2;

PBK = pihat_sorted2(pihat_sorted2(:,1)==1,:);  %and this
PWG = pihat_sorted2(pihat_sorted2(:,1)==2,:);
PBG = pihat_sorted2(pihat_sorted2(:,1)==3,:);
POT = pihat_sorted2(pihat_sorted2(:,1)==4,:);

PBK_sorted = sortrows(PBK,[3 4 5 6]);PBK_sorted = flipud(PBK_sorted);
PWG_sorted = sortrows(PWG,[4 3 5 6]);PWG_sorted = flipud(PWG_sorted);
PBG_sorted = sortrows(PBG,[5 3 4 6]);PBG_sorted = flipud(PBG_sorted);
POT_sorted = sortrows(POT,[6 3 4 6]);POT_sorted = flipud(POT_sorted);

PBK_NAIVE_sorted = PBK_sorted(PBK_sorted(:,2)==0,3:6); 
PWG_NAIVE_sorted = PWG_sorted(PWG_sorted(:,2)==0,3:6); 
PBG_NAIVE_sorted = PBG_sorted(PBG_sorted(:,2)==0,3:6); 
POT_NAIVE_sorted = POT_sorted(POT_sorted(:,2)==0,3:6); 

PBK_NOTNAIVE_sorted = PBK_sorted(PBK_sorted(:,2)==1,3:6); 
PWG_NOTNAIVE_sorted = PWG_sorted(PWG_sorted(:,2)==1,3:6); 
PBG_NOTNAIVE_sorted = PBG_sorted(PBG_sorted(:,2)==1,3:6); 
POT_NOTNAIVE_sorted = POT_sorted(POT_sorted(:,2)==1,3:6); 

PBK_sorted = PBK_sorted(:,3:6); 
PWG_sorted = PWG_sorted(:,3:6); 
PBG_sorted = PBG_sorted(:,3:6); 
POT_sorted = POT_sorted(:,3:6); 

pihat_sorted2 = [PBK_NAIVE_sorted;PBK_NOTNAIVE_sorted;...
    PWG_NAIVE_sorted;PWG_NOTNAIVE_sorted;...
    PBG_NAIVE_sorted;PBG_NOTNAIVE_sorted;...
    POT_NAIVE_sorted;POT_NOTNAIVE_sorted];

pihat_sorted_NAIVE = [PBK_NAIVE_sorted;...
    PWG_NAIVE_sorted;...
    PBG_NAIVE_sorted;...
    POT_NAIVE_sorted];

pihat_sorted_NOTNAIVE = [PBK_NOTNAIVE_sorted;...
    PWG_NOTNAIVE_sorted;...
    PBG_NOTNAIVE_sorted;...
    POT_NOTNAIVE_sorted];


else
    [pihat,dlow,dhi] = mnrval(B,LUV_TRS,stats);%,'model','hierarchical','type','conditional');
    [pihat_NAIVE,dlow_NAIVE,dhi_NAIVE] = mnrval(B,NAIVE_LUV_TRS,stats);
    [pihat_NOTNAIVE,dlow_NOTNAIVE,dhi_NOTNAIVE] = mnrval(B,NOTNAIVE_LUV_TRS,stats);
    %find the distribution of bk, wg, bb other predicted in the training out data:
pihat_binarized_TRS=[];
pihat_NAIVE_binarized_TRS=[];
pihat_NOTNAIVE_binarized_TRS=[];
for pih = 1:size(pihat,1)
    pihat_binarized_TRS = [pihat_binarized_TRS; pihat(pih,:) == max(pihat(pih,:))];
end
for pih = 1:size(pihat_NAIVE,1)
    pihat_NAIVE_binarized_TRS = [pihat_NAIVE_binarized_TRS; pihat_NAIVE(pih,:) == max(pihat_NAIVE(pih,:))];
end
for pih = 1:size(pihat_NOTNAIVE,1)
    pihat_NOTNAIVE_binarized_TRS = [pihat_NOTNAIVE_binarized_TRS; pihat_NOTNAIVE(pih,:) == max(pihat_NOTNAIVE(pih,:))];
end
if binarize==1
PIHAT_DIST_TRS = sum(pihat_binarized_TRS);
VERBAL_DIST_TRS = [numBK_TRS numWG_TRS numBG_TRS numOT_TRS];
PIHAT_NAIVE_DIST_TRS = sum(pihat_NAIVE_binarized_TRS);
VERBAL_NAIVE_DIST_TRS = [NAIVE_numBK_TRS NAIVE_numWG_TRS NAIVE_numBG_TRS NAIVE_numOT_TRS];
PIHAT_NOTNAIVE_DIST_TRS = sum(pihat_NOTNAIVE_binarized_TRS);
VERBAL_NOTNAIVE_DIST_TRS = [NOTNAIVE_numBK_TRS NOTNAIVE_numWG_TRS NOTNAIVE_numBG_TRS NOTNAIVE_numOT_TRS];
else
PIHAT_DIST_TRS = sum(pihat);
VERBAL_DIST_TRS = [numBK_TRS numWG_TRS numBG_TRS numOT_TRS];
PIHAT_NAIVE_DIST_TRS = sum(pihat_NAIVE);
VERBAL_NAIVE_DIST_TRS = [NAIVE_numBK_TRS NAIVE_numWG_TRS NAIVE_numBG_TRS NAIVE_numOT_TRS];
PIHAT_NOTNAIVE_DIST_TRS = sum(pihat_NOTNAIVE);
VERBAL_NOTNAIVE_DIST_TRS = [NOTNAIVE_numBK_TRS NOTNAIVE_numWG_TRS NOTNAIVE_numBG_TRS NOTNAIVE_numOT_TRS];
DHI_TRS = sum(dhi)./sum(sum(pihat))*100;
DHI_NAIVE_TRS = sum(dhi_NAIVE)./sum(sum(pihat_NAIVE))*100;
DHI_NOTNAIVE_TRS = sum(dhi_NOTNAIVE)./sum(sum(pihat_NOTNAIVE))*100;
end

    
output=[CATS_TRS NOTNAIVE_TRS pihat]; output_sorted = sortrows(output,[1 2]);
pihat_sorted2 = output_sorted;%(:,3:6);
CATS_TRS_sorted2=output_sorted(:,1);
CATS_sorted2=CATS_TRS_sorted2;

PBK = pihat_sorted2(pihat_sorted2(:,1)==1,:);  %and this
PWG = pihat_sorted2(pihat_sorted2(:,1)==2,:);
PBG = pihat_sorted2(pihat_sorted2(:,1)==3,:);
POT = pihat_sorted2(pihat_sorted2(:,1)==4,:);

PBK_sorted = sortrows(PBK,3);PBK_sorted = flipud(PBK_sorted);
PWG_sorted = sortrows(PWG,4);PWG_sorted = flipud(PWG_sorted);
PBG_sorted = sortrows(PBG,5);PBG_sorted = flipud(PBG_sorted);
POT_sorted = sortrows(POT,6);POT_sorted = flipud(POT_sorted);

PBK_NAIVE_sorted = PBK_sorted(PBK_sorted(:,2)==0,3:6); 
PWG_NAIVE_sorted = PWG_sorted(PWG_sorted(:,2)==0,3:6); 
PBG_NAIVE_sorted = PBG_sorted(PBG_sorted(:,2)==0,3:6); 
POT_NAIVE_sorted = POT_sorted(POT_sorted(:,2)==0,3:6); 

PBK_NOTNAIVE_sorted = PBK_sorted(PBK_sorted(:,2)==1,3:6); 
PWG_NOTNAIVE_sorted = PWG_sorted(PWG_sorted(:,2)==1,3:6); 
PBG_NOTNAIVE_sorted = PBG_sorted(PBG_sorted(:,2)==1,3:6); 
POT_NOTNAIVE_sorted = POT_sorted(POT_sorted(:,2)==1,3:6); 

PBK_sorted = PBK_sorted(:,3:6);
PWG_sorted = PWG_sorted(:,3:6);
PBG_sorted = PBG_sorted(:,3:6);
POT_sorted = POT_sorted(:,3:6);

pihat_sorted2 = [PBK_NAIVE_sorted;PBK_NOTNAIVE_sorted;...
    PWG_NAIVE_sorted;PWG_NOTNAIVE_sorted;...
    PBG_NAIVE_sorted;PBG_NOTNAIVE_sorted;...
    POT_NAIVE_sorted;POT_NOTNAIVE_sorted];

pihat_sorted_NAIVE = [PBK_NAIVE_sorted;...
    PWG_NAIVE_sorted;...
    PBG_NAIVE_sorted;...
    POT_NAIVE_sorted];

pihat_sorted_NOTNAIVE = [PBK_NOTNAIVE_sorted;...
    PWG_NOTNAIVE_sorted;...
    PBG_NOTNAIVE_sorted;...
    POT_NOTNAIVE_sorted];

end

Cs={'B/K','W/G','B/B','Other'};
PM=[];
YM=[];
for c=1:4
% figure(11+t)
% subplot(1,4,c)
    if test_left_out == 1
        cats = CATS_TES;
    else
        cats = CATS_TRS;
    end

%boxplot(pihat(:,c)',cats)
% 
%     ax = gca;
%     ax.Box='off';
%     ax.LineWidth = 1.5;
%     ax.XTick = [1 2 3 4];
%     ax.XTickLabel = {'B/K','W/G','B/B','OT'};
%     xlabel('Verbal Report')
%     %ylabel('Predicted probability of category membership')
%    title(['Predicted probability ' Cs(c) ' '])
%     ax.FontSize = 12;
%     xlim([0 5]) 
    PM=[PM; mean(pihat(cats==c,:))];
% 
% set(11+t,'color',[1 1 1])
% figure(1+t)
% subplot(1,4,c)
%hist(stats.resid(:,c))
%title([' residuals  ' Cs(c) 'category'])
end



%figure(13+t)
%I=imagesc(pihat_sorted2');
%colorbar
% caxis([0 1])
% ax = gca;
%     ax.Box='off';
%     ax.LineWidth = 1.5;
%     ax.YTick = [1 2 3 4];
%     ax.YTickLabel = {'B/K','W/G','B/B','OT'};
    G=[];
    for c = 1:4
        g=sum(CATS_sorted2==c);
        G=[G g];
    end
    %    hold on
    
     Gs = [ G(1) (G(2)+G(1)+1),...
         (G(3)+G(1)+(G(2)+1))];
   %  stem(Gs,ones(3,1)*10,':w','LineWidth',2)
%     title('Predicted probability of category membership, given matches')
%     ax.FontSize = 20;
%     set(13+t,'color',[1 1 1])
%     hold off
%      if save_figs == 1
%          if test_left_out == 1
%         savefig([file_path '_HEAT_SCALE_LEFTOUTDATA'])
%          elseif test_left_out == 0
%         savefig([file_path '_HEAT_SCALE_TRAININGDATA'])
%          end
%     end
    
      
    if test_left_out == 1
        what_data = {'left out data'};
    else
        what_data = {'training data'};
    end
%         figure(17+t)
%         subplot(2,1,1)
%         I=imagesc(pihat_sorted_NAIVE');
%         colorbar
%         caxis([0 1])
%         ax = gca;
%         ax.Box='off';
%         ax.LineWidth = 1.5;
%         ax.YTick = [1 2 3 4];
%         ax.YTickLabel = {'B/K','W/G','B/B','OT'};
%         title(['NAIVE - '  what_data])
%         ax.FontSize = 16;
%         set(17+t,'color',[1 1 1])
        
%         subplot(2,1,2)
%         I=imagesc(pihat_sorted_NOTNAIVE');
%         colorbar
%         caxis([0 1])
%         ax = gca;
%         ax.Box='off';
%         ax.LineWidth = 1.5;
%         ax.YTick = [1 2 3 4];
%         ax.YTickLabel = {'B/K','W/G','B/B','OT'};
%         title(['NOT NAIVE - ' what_data])
%         ax.FontSize = 16;
%         set(17+t,'color',[1 1 1])
%           if save_figs == 1
%          if test_left_out == 1
%         savefig([file_path '_HEAT_SCALE_NAIVEorNOT_LEFTOUTDATA'])
%          elseif test_left_out == 0
%         savefig([file_path '_HEAT_SCALE_NAIVEorNOT_TRAININGDATA'])
%          end
%     end
%     
        
    
% figure(16+t)
% subplot(2,2,1)
% I=imagesc(PBK_sorted')
% title('B/K reporters')
% colorbar;caxis([0 1]) %make colorscale bar go from 0 to 1
% ax = gca;ax.Box='off';ax.LineWidth = 1.5;
% ax.YTick = [1 2 3 4];ax.YTickLabel = {'B/K','W/G','B/B','OT'};
% ax.FontSize = 16;set(16+t,'color',[1 1 1]); hold on 
% 
% subplot(2,2,2)
% I=imagesc(PWG_sorted')
% title('W/G reporters')
% colorbar;caxis([0 1])
% ax = gca;ax.Box='off';ax.LineWidth = 1.5;
% ax.YTick = [1 2 3 4];ax.YTickLabel = {'B/K','W/G','B/B','OT'};
% ax.FontSize = 16;set(16+t,'color',[1 1 1]); hold on 
%  
% subplot(2,2,3)
% I=imagesc(PBG_sorted');
% title('B/B reporters')
% colorbar;caxis([0 1])
% ax = gca;ax.Box='off';ax.LineWidth = 1.5;
% ax.YTick = [1 2 3 4];ax.YTickLabel = {'B/K','W/G','B/B','OT'};
% ax.FontSize = 16;set(16+t,'color',[1 1 1]); hold on 
%   
% subplot(2,2,4)
% I=imagesc(POT_sorted');
% title('Other reporters')
% colorbar;caxis([0 1])
% ax = gca;ax.Box='off';ax.LineWidth = 1.5;
% ax.YTick = [1 2 3 4];ax.YTickLabel = {'B/K','W/G','B/B','OT'};
% ax.FontSize = 16;set(16+t,'color',[1 1 1]);
% hold on 
% if save_figs == 1
%     if test_left_out == 1
%         savefig([file_path '_HEAT_SCALE_byGroup_LEFTOUTDATA'])
%     elseif test_left_out == 0
%         savefig([file_path '_HEAT_SCALE_byGroup_TRAININGDATA'])
%     end
% end


figure(3-t)
subplot(4,2,1)
I=imagesc(PBK_NAIVE_sorted')
title('B/K NAIVE')
colorbar;caxis([0 1]) %make colorscale bar go from 0 to 1
ax = gca;ax.Box='off';ax.LineWidth = 1.5;
ax.YTick = [1 2 3 4];ax.YTickLabel = {'B/K','W/G','B/B','OT'};
ax.FontSize = 16;set(3-t,'color',[1 1 1]); hold on 
subplot(4,2,2)
I=imagesc(PBK_NOTNAIVE_sorted')
title('B/K NOT NAIVE')
colorbar;caxis([0 1]) %make colorscale bar go from 0 to 1
ax = gca;ax.Box='off';ax.LineWidth = 1.5;
ax.YTick = [1 2 3 4];ax.YTickLabel = {'B/K','W/G','B/B','OT'};
ax.FontSize = 16;set(3-t,'color',[1 1 1]); hold on 
subplot(4,2,3)
I=imagesc(PWG_NAIVE_sorted')
title('W/G NAIVE')
colorbar;caxis([0 1])
ax = gca;ax.Box='off';ax.LineWidth = 1.5;
ax.YTick = [1 2 3 4];ax.YTickLabel = {'B/K','W/G','B/B','OT'};
ax.FontSize = 16;set(3-t,'color',[1 1 1]); hold on 
subplot(4,2,4)
I=imagesc(PWG_NOTNAIVE_sorted')
title('W/G NOT NAIVE')
colorbar;caxis([0 1])
ax = gca;ax.Box='off';ax.LineWidth = 1.5;
ax.YTick = [1 2 3 4];ax.YTickLabel = {'B/K','W/G','B/B','OT'};
ax.FontSize = 16;set(3-t,'color',[1 1 1]); hold on 
 
subplot(4,2,5)
I=imagesc(PBG_NAIVE_sorted');
title('B/B NAIVE')
colorbar;caxis([0 1])
ax = gca;ax.Box='off';ax.LineWidth = 1.5;
ax.YTick = [1 2 3 4];ax.YTickLabel = {'B/K','W/G','B/B','OT'};
ax.FontSize = 16;set(3-t,'color',[1 1 1]); hold on 
subplot(4,2,6)
I=imagesc(PBG_NOTNAIVE_sorted');
title('B/B NOT NAIVE')
colorbar;caxis([0 1])
ax = gca;ax.Box='off';ax.LineWidth = 1.5;
ax.YTick = [1 2 3 4];ax.YTickLabel = {'B/K','W/G','B/B','OT'};
ax.FontSize = 16;set(3-t,'color',[1 1 1]); hold on 
 
subplot(4,2,7)
I=imagesc(POT_NAIVE_sorted');
title('Other NAIVE')
colorbar;caxis([0 1])
ax = gca;ax.Box='off';ax.LineWidth = 1.5;
ax.YTick = [1 2 3 4];ax.YTickLabel = {'B/K','W/G','B/B','OT'};
ax.FontSize = 16;set(3-t,'color',[1 1 1]);hold on 
xlabel('Subject')
subplot(4,2,8)
I=imagesc(POT_NOTNAIVE_sorted');
title('Other NOT NAIVE')
colorbar;caxis([0 1])
ax = gca;ax.Box='off';ax.LineWidth = 1.5;
ax.YTick = [1 2 3 4];ax.YTickLabel = {'B/K','W/G','B/B','OT'};
ax.FontSize = 16;set(3-t,'color',[1 1 1]);hold on 
xlabel('Subject')
set(gcf, 'Position', [474, 69, 591, 886])

 
% if save_figs == 1
%     if test_left_out == 1
%         savefig([file_path '_HEAT_SCALE_byGroup_LEFTOUTDATA'])
%     elseif test_left_out == 0
%         savefig([file_path '_HEAT_SCALE_byGroup_TRAININGDATA'])
%     end
% end



 test_left_out=1;

 
 
 
end
 close Figure 2

    numTES = numBK_TES+numWG_TES+numBG_TES+numOT_TES;
    numTRS = numBK_TRS+numWG_TRS+numBG_TRS+numOT_TRS;
DISTS= [PIHAT_DIST_TRS/numTRS*100;VERBAL_DIST_TRS/numTRS*100;...
    PIHAT_DIST_TES/numTES*100;VERBAL_DIST_TES/numTES*100];

if binarize==0
DHIS=[DHI_TRS;0 0 0 0;DHI_TES;0 0 0 0];
DHIS_NAIVE = [DHI_NAIVE_TRS; 0 0 0 0; DHI_NAIVE_TES; 0 0 0 0];
DHIS_NOTNAIVE = [DHI_NOTNAIVE_TRS; 0 0 0 0; DHI_NOTNAIVE_TES; 0 0 0 0];
end

  NAIVE_numTES = NAIVE_numBK_TES+NAIVE_numWG_TES+NAIVE_numBG_TES+NAIVE_numOT_TES;
    NAIVE_numTRS = NAIVE_numBK_TRS+NAIVE_numWG_TRS+NAIVE_numBG_TRS+NAIVE_numOT_TRS;
NAIVE_DISTS= [PIHAT_NAIVE_DIST_TRS/NAIVE_numTRS*100;VERBAL_NAIVE_DIST_TRS/NAIVE_numTRS*100;...
    PIHAT_NAIVE_DIST_TES/NAIVE_numTES*100;VERBAL_NAIVE_DIST_TES/NAIVE_numTES*100];

  NOTNAIVE_numTES = NOTNAIVE_numBK_TES+NOTNAIVE_numWG_TES+NOTNAIVE_numBG_TES+NOTNAIVE_numOT_TES;
    NOTNAIVE_numTRS = NOTNAIVE_numBK_TRS+NOTNAIVE_numWG_TRS+NOTNAIVE_numBG_TRS+NOTNAIVE_numOT_TRS;
NOTNAIVE_DISTS= [PIHAT_NOTNAIVE_DIST_TRS/NOTNAIVE_numTRS*100;VERBAL_NOTNAIVE_DIST_TRS/NOTNAIVE_numTRS*100;...
    PIHAT_NOTNAIVE_DIST_TES/NOTNAIVE_numTES*100;VERBAL_NOTNAIVE_DIST_TES/NOTNAIVE_numTES*100];

%    classout=zeros(length(CATS_TES),1);
%    truelabels = CATS_TES;
%    classout(pihat_binarized_TES(:,1)==1) = 1;
%    classout(pihat_binarized_TES(:,2)==1) = 2;
%     classout(pihat_binarized_TES(:,3)==1) = 3;
%      classout(pihat_binarized_TES(:,4)==1) = 4;
%       main_cat = [1 2]; 
%     sec_cat = [3 4];
%     CP_TES = classperf(truelabels, classout)
%    CP_TES_main_sec = classperf(truelabels, classout, 'Positive', main_cat, 'Negative', sec_cat)
%   %https://www.mathworks.com/help/bioinfo/ref/classperf.html
%    CP_TES_bg = classperf(truelabels, classout, 'Positive', 3, 'Negative', [1 2 4])
%     CP_TES_bk = classperf(truelabels, classout, 'Positive', 1, 'Negative', [2 3 4])
%     CP_TES_wg = classperf(truelabels, classout, 'Positive', 2, 'Negative', [1 3 4])
%     CP_TES_ot = classperf(truelabels, classout, 'Positive', 4, 'Negative', [1 2 3])
%  
%   
%  classout=zeros(length(NAIVE_CATS_TES),1);
%    truelabels = NAIVE_CATS_TES;
%    classout(pihat_NAIVE_binarized_TES(:,1)==1) = 1;
%    classout(pihat_NAIVE_binarized_TES(:,2)==1) = 2;
%     classout(pihat_NAIVE_binarized_TES(:,3)==1) = 3;
%      classout(pihat_NAIVE_binarized_TES(:,4)==1) = 4;
%       main_cat = [1 2];
%     sec_cat = [3 4];
%     
%     CP_NAIVE = classperf(truelabels, classout)
%     CP_NAIVE_main_sec = classperf(truelabels, classout, 'Positive', main_cat, 'Negative', sec_cat)
%     CP_NAIVE_bg = classperf(truelabels, classout, 'Positive', 3, 'Negative', [1 2 4])
%    CP_NAIVE_bk = classperf(truelabels, classout, 'Positive', 1, 'Negative', [2 3 4])
%     CP_NAIVE_wg = classperf(truelabels, classout, 'Positive', 2, 'Negative', [1 3 4])
%     CP_NAIVE_ot = classperf(truelabels, classout, 'Positive', 4, 'Negative', [1 2 3])
%  
%     
%   classout=zeros(length(NOTNAIVE_CATS_TES),1);
%    truelabels = NOTNAIVE_CATS_TES;
%    classout(pihat_NOTNAIVE_binarized_TES(:,1)==1) = 1;
%    classout(pihat_NOTNAIVE_binarized_TES(:,2)==1) = 2;
%     classout(pihat_NOTNAIVE_binarized_TES(:,3)==1) = 3;
%      classout(pihat_NOTNAIVE_binarized_TES(:,4)==1) = 4;
%       main_cat = [1 2];
%     sec_cat = [3 4];
%     CP_NOTNAIVE = classperf(truelabels, classout)
%     CP_NOTNAIVE_main_sec = classperf(truelabels, classout, 'Positive', main_cat, 'Negative', sec_cat)
%    CP_NOTNAIVE_bg = classperf(truelabels, classout, 'Positive', 3, 'Negative', [1 2 4])
%  CP_NOTNAIVE_bk = classperf(truelabels, classout, 'Positive', 1, 'Negative', [2 3 4])
%     CP_NOTNAIVE_wg = classperf(truelabels, classout, 'Positive', 2, 'Negative', [1 3 4])
%     CP_NOTNAIVE_ot = classperf(truelabels, classout, 'Positive', 4, 'Negative', [1 2 3])
%  
%    
% %bar plot of sensitivity metrics 
%  sensitivity_naive = [CP_NAIVE.CorrectRate CP_NAIVE_bk.Sensitivity 	 CP_NAIVE_wg.Sensitivity  CP_NAIVE_bg.Sensitivity	 CP_NAIVE_ot.Sensitivity];   
%   sensitivity_notnaive = [CP_NOTNAIVE.CorrectRate CP_NOTNAIVE_bk.Sensitivity 	 CP_NOTNAIVE_wg.Sensitivity CP_NOTNAIVE_bg.Sensitivity	 CP_NOTNAIVE_ot.Sensitivity];   
% 
% %  figure(31);
% %  subplot(1,2,1)
% %  bar(sensitivity_naive)
% %  ylabel('Accuracy (Correctly Classified Positive Samples / True Positive Samples)')
% %  ax = gca;ax.Box='off';ax.LineWidth = 1.5;ax.XTick = [1 2 3 4 5];ax.FontSize = 16;
% % ax.XTickLabel = {'All','B/K','W/G','B/B','OT'}; 
% % title('Naive')
% %  subplot(1,2,2)
% %  bar(sensitivity_notnaive)
% %   ax = gca;ax.Box='off';ax.LineWidth = 1.5;ax.XTick = [1 2 3 4 5];ax.FontSize = 16;
% % ax.XTickLabel = {'All','B/K','W/G','B/B','OT'}; 
% % title('Not Naive')
% % 
% 
% 
meanP_hits_naive_bk = mean(pihat_NAIVE((pihat_NAIVE_binarized_TES(:,1)==1)&(NAIVE_CATS_TES==1),1))
meanP_hits_naive_wg = mean(pihat_NAIVE((pihat_NAIVE_binarized_TES(:,2)==1)&(NAIVE_CATS_TES==2),2))
meanP_hits_naive_bg = mean(pihat_NAIVE((pihat_NAIVE_binarized_TES(:,3)==1)&(NAIVE_CATS_TES==3),3))
meanP_hits_naive_ot = mean(pihat_NAIVE((pihat_NAIVE_binarized_TES(:,4)==1)&(NAIVE_CATS_TES==4),4))

meanP_hits_notnaive_bk = mean(pihat_NOTNAIVE((pihat_NOTNAIVE_binarized_TES(:,1)==1)&(NOTNAIVE_CATS_TES==1),1))
meanP_hits_notnaive_wg = mean(pihat_NOTNAIVE((pihat_NOTNAIVE_binarized_TES(:,2)==1)&(NOTNAIVE_CATS_TES==2),2))
meanP_hits_notnaive_bg = mean(pihat_NOTNAIVE((pihat_NOTNAIVE_binarized_TES(:,3)==1)&(NOTNAIVE_CATS_TES==3),3))
meanP_hits_notnaive_ot = mean(pihat_NOTNAIVE((pihat_NOTNAIVE_binarized_TES(:,4)==1)&(NOTNAIVE_CATS_TES==4),4))

meanP_hits_bk = mean(pihat((pihat_binarized_TES(:,1)==1)&(CATS_TES==1),1))
meanP_hits_wg = mean(pihat((pihat_binarized_TES(:,2)==1)&(CATS_TES==2),2))
meanP_hits_bg = mean(pihat((pihat_binarized_TES(:,3)==1)&(CATS_TES==3),3))
meanP_hits_ot = mean(pihat((pihat_binarized_TES(:,4)==1)&(CATS_TES==4),4))

meanP_bk_reporters = mean(pihat((CATS_TES==1),1))
meanP_wg_reporters = mean(pihat((CATS_TES==2),2))
meanP_bg_reporters = mean(pihat((CATS_TES==3),3))
meanP_ot_reporters = mean(pihat((CATS_TES==4),4))

meanP_bkhit_bg = mean(pihat((pihat_binarized_TES(:,1)==1)&(CATS_TES==3),1))
meanP_wghit_bg = mean(pihat((pihat_binarized_TES(:,2)==1)&(CATS_TES==3),2))

% 

