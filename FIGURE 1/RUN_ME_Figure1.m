%%%This script generates the components of Figure 1 from the publication: 
% Lafer-Sousa, R., & Conway, B. R. (2017). #thedress: Categorical perception of an ambiguous color image. Journal of Vision
% Figure legend:
% Figure 1. Population distributions of subjects? color matches show categorical perception of the dress. Subjects used a digital color
% picker to match their perception of four regions of the dress (i, ii, iii, iv); the dress image was shown throughout the color-matching
% procedure. (A) Matches for regions i and iv of the dress plotted against matches for regions ii and iii, for all online subjects (N=2,200;
% R = 0.62, p , 0.001). Contours contain the highest density (75%) of matches. The first principal component of the population
% matches (computed from CIELUV values) to (i, iv) defined the y-axis (gold/black: GK); the first principal component of the population
% matches to (ii, iii) defined the x-axis (white/blue: WB). Each subject?s (x, y) values are the principal-component weights for their
% matches; each has two (x, y) pairs, corresponding to (i, ii) and (iii, iv). Color scale is number of matches (smoothed). (B) Color matches
% for regions (i, iii) of the dress plotted against matches for regions (ii, iv) for subjects who had never seen the dress before the
% experiment (Na¨?ve; N = 1,017; R = 0.62, p , 0.001). Axes and contours were defined using data from only those subjects. (C) Color
% matches for regions (i, iv) of the dress plotted against matches for regions (ii, iii) for subjects who had seen the dress before the
% experiment (N = 1,183; R = 0.61, p , 0.001). Axes and contours were defined using data from only those subjects. (D) Color matches
% for all subjects (from A) were sorted by subjects? verbal color descriptions (??blue/black??=B/K, N=1,184; ??white/gold??=W/G, N=686; 
% ??blue/brown??= B/B, N=272) and plotted separately. Axes defined as in (A). In all panels, contours contain the highest density
% (75%) of the matches shown in each plot. Dress image reproduced with permission from Cecilia Bleasdale.

%%%The script was written by Rosa Lafer-Sousa. (for questions, contact: rlaferso@mit.edu)
    % Note: this code was modified from its original format to produce only the contents of Figure 1 from the publication. last
    % modified: 10/13/2017.  Matlab version 2016B (originally written in with Matlab 2015B)

%%%Notes on running the code:
% The code calls on three support scripts, provided in the present directory:
        %Luv2RGB.m
        %scatplot_RLS_2.m
        %scatplot_RLS_CONTOUR.m
% The code loads data from several .mat files that were generated from the raw data provided in the folder RAW DATA:
    %All the data we collected online (Amazon's Mechanical Turk) from the 'SCALE Experiment' and 'MAIN Experiment' (all scales presented); N = 2200:
        %  TURK_MAIN_and_SCALE1036100150_BK_ALL.mat   'Blue/Black' reporters
        %  TURK_MAIN_and_SCALE1036100150_WG_ALL.mat   'White/Gold' reporters
        %  TURK_MAIN_and_SCALE1036100150_BG_ALL.mat   'Blue/Brown' reporters
        %  TURK_MAIN_and_SCALE1036100150_OT_ALL.mat   'Other color' reporters
    %The subset of online data correspoding to subjects who HAD NOT seen the Dress image prior to taking part in our study:
        %  TURK_MAIN_and_SCALE1036100150_BK_NAIVE.mat   'Blue/Black' reporters
        %  TURK_MAIN_and_SCALE1036100150_WG_NAIVE.mat   'White/Gold' reporters
        %  TURK_MAIN_and_SCALE1036100150_BG_NAIVE.mat   'Blue/Brown' reporters
        %  TURK_MAIN_and_SCALE1036100150_OT_NAIVE.mat   'Other color' reporters
    %The subset of online data correspoding to subjects who HAD seen the Dress image prior to taking part in our study:
        %  TURK_MAIN_and_SCALE1036100150_BK_EXP.mat   'Blue/Black' reporters
        %  TURK_MAIN_and_SCALE1036100150_WG_EXP.mat   'White/Gold' reporters
        %  TURK_MAIN_and_SCALE1036100150_BG_EXP.mat   'Blue/Brown' reporters
        %  TURK_MAIN_and_SCALE1036100150_OT_EXP.mat   'Other color' reporters
        
% Running this script will generate three figure files:
   %Figure 1 - Desity scatter plots shown in panels A-D of the published Figure 
   %Figure 2 - Provides the Principal Component color axes that correspond to the axes of the scatter plots shown in Figure 1. 
   %Figure 3 - 'the Dress' image used in the study with arrows indicating the regions of the dress to which subjects made their color matches.

 
clear all; close all; clc


dsets = [19 20 21];
for pop = 1:3 
    dataset = dsets(pop);
 
top_P = 75;  %;25 50 80 set to perent of match points for density contours
regions_up_low = 3; %set to 1 for upper dress regions, 2 for lower; 3 for both


if dataset == 19
      filenames =   {'TURK_MAIN_and_SCALE1036100150_BK_EXP';...
       'TURK_MAIN_and_SCALE1036100150_WG_EXP';...
        'TURK_MAIN_and_SCALE1036100150_BG_EXP';...
        'TURK_MAIN_and_SCALE1036100150_OT_EXP'};%experienced subjects from the SCALE expt and MAIN turk (all scales presented) N = 

elseif dataset == 20
      filenames =   {'TURK_MAIN_and_SCALE1036100150_BK_NAIVE';...
       'TURK_MAIN_and_SCALE1036100150_WG_NAIVE';...
        'TURK_MAIN_and_SCALE1036100150_BG_NAIVE';...
        'TURK_MAIN_and_SCALE1036100150_OT_NAIVE'};%naive subjects from the SCALE expt and MAIN turk (all scales presented) N = 
model_filename='NAIVE_lin_reg'; 
 elseif dataset == 21
      filenames =   {'TURK_MAIN_and_SCALE1036100150_BK_ALL'...
       'TURK_MAIN_and_SCALE1036100150_WG_ALL'...
        'TURK_MAIN_and_SCALE1036100150_BG_ALL'...
        'TURK_MAIN_and_SCALE1036100150_OT_ALL'};%ALL subjects from the SCALE expt and MAIN turk (all scales presented) N = 2200
  
end 


LUV_U_BY_REGION = [];
LUV_V_BY_REGION = [];
LUV_L_BY_REGION = [];
RGB_A=[];
RGB_B=[];
RGB_C=[];
RGB_D=[];
U_RAND_G1 = [];
U_RAND_B1 = [];
U_RAND_G2 = [];
U_RAND_B2 = [];
V_RAND_G1 = [];
V_RAND_B1 = [];
V_RAND_G2 = [];
V_RAND_B2 = [];
L_RAND_G1 = [];
L_RAND_B1 = [];
L_RAND_G2 = [];
L_RAND_B2 = [];
GROUP=[];


%%combine the data sets:
for percept = 1:length(filenames)
    load(filenames{percept})
    LUV_U_BY_REGION = [LUV_U_BY_REGION; Luv_u_by_region];
    LUV_V_BY_REGION = [LUV_V_BY_REGION; Luv_v_by_region];
    LUV_L_BY_REGION = [LUV_L_BY_REGION; Luv_L_by_region];
    RGB_A = [RGB_A; resp_A];
    RGB_B = [RGB_B; resp_B];
    RGB_C = [RGB_C; resp_C];
    RGB_D = [RGB_D; resp_D];
    group = percept*(ones(size(Luv_u_by_region,1),1));
    GROUP = [GROUP; group];
    
end



GROUP2 = [GROUP;GROUP]; %need this for when plotting upper and lower dress region matches (b1b2 vs g1g2), as this makes for twice as many data points)
Luv_u_by_region = LUV_U_BY_REGION;
Luv_v_by_region = LUV_V_BY_REGION;
Luv_L_by_region = LUV_L_BY_REGION;

u = Luv_u_by_region;
v = Luv_v_by_region;
L = Luv_L_by_region;



%%---RUN PCA--------------------------------------------------------:

g1=[u(:,1),v(:,1),L(:,1)];
g2=[u(:,4),v(:,4),L(:,4)];
b1=[u(:,2),v(:,2),L(:,2)];
b2=[u(:,3),v(:,3),L(:,3)];

if regions_up_low == 1
    b12 = b1';
    g12 = g1';
elseif regions_up_low == 2
    b12 = b2';
    g12 = g2';
    elseif regions_up_low == 3
    b12 = [b1' b2'];
    g12 = [g1' g2'];
    GROUP=GROUP2;
end


if regions_up_low == 1
    blabel = 'PC1, WB_t';
    glabel = 'PC1, GK_t';
elseif regions_up_low == 2
     blabel = 'PC1, WB_b';
    glabel = 'PC1, GK_b';
    elseif regions_up_low == 3
     blabel = 'PC1, WB';
    glabel = 'PC1, GK'; 
end

%PCA starts here:

%%

% [[3 x alot], [3 x alot]], [3 x 2*alot]
%IN THIS CODE the variable 'B12' could = B1 or B2 or B12 (same goes for G12) (it depends on how you set the variable regions_up_low at the start)
col_grandmean_b12 = mean(b12,2);
b12_demean = nan(size(b12));
for i = 1:3
    b12_demean(i,:) = b12(i,:) - mean(b12(i,:));
end

% PCA
[U, S, V] = svd(b12_demean, 'econ');

% eigenvalues
eigvals = diag(S).^2;
eigvals = eigvals/sum(eigvals);

% first component
b12_pca_first_comp = U(:,1)*S(1,1);
b12_pca_first_comp_weights = V(:,1)';

% reconstructed colors
luv_b12_orig = b12_pca_first_comp * b12_pca_first_comp_weights + repmat(col_grandmean_b12, [1, length(b12_pca_first_comp_weights)]);

%this lets you evenly sample the vector, rather than just sampling at sites where there are data points:
min_b12 = min(b12_pca_first_comp_weights);
max_b12 = max(b12_pca_first_comp_weights);
b12_vect_points = linspace(min_b12,max_b12,1000);
luv_b12 = b12_pca_first_comp * b12_vect_points + repmat(col_grandmean_b12, [1, length(b12_vect_points)]);

luvim_b12 = zeros(size(luv_b12,2),1,3); %initialize the matrix for the best fit line RGB vals
for i = 1:size(luv_b12,2)
    luvim_b12(i,1,2) = luv_b12(1,i);    %function expects L in first dim, u in sepercept, y in third
    luvim_b12(i,1,3) = luv_b12(2,i);   %function expects L in first dim, u in sepercept, y in third
    luvim_b12(i,1,1) = luv_b12(3,i);  %function expects L in first dim, u in sepercept, y in third
end
rgb_b12 = squeeze(Luv2RGB(luvim_b12))';

min_bin_b12 = min(b12_pca_first_comp_weights);
max_bin_b12 = max(b12_pca_first_comp_weights);

rgb_b12 = rgb_b12';
Ib12 = ones(10,1000,3);

for i = 1:1000
   
Ib12(:,i,1)=rgb_b12(i,1);
Ib12(:,i,2)=rgb_b12(i,2);
Ib12(:,i,3)=rgb_b12(i,3);

end


%plot the color scale bars for the PC axes
hpanel = [1 2 3];
cpanel = [4 5 6];

figure(2)
subplot(4,3,hpanel(pop))
hist(b12_pca_first_comp_weights,100)
xlim([min_bin_b12 max_bin_b12])
xlabel('PC WB')
ylabel('# of color matches')
if dataset ==21
   title('All Subjects')
elseif dataset ==20
    title('Naive Subjects')
elseif dataset ==19
    title('Not-Naive Subjects')
end
%  map = rgb_b12;
%  cmap = colormap(map);
%  cmapb = cmap;
%  colormap(cmapb)
% h = colorbar;
% set(h,'XTickLabel',{})
% set(h,'location','SouthOutside')
subplot(4,3,cpanel(pop))
Ib12s = [Ib12;Ib12;Ib12;Ib12;Ib12;Ib12;Ib12;Ib12];
imshow(Ib12s);
set(gcf,'color','w');
set(gcf, 'Position', [1, 1, 1111, 570])



%%%%NOW DO GOLD/BLACK REGION OF DRESS --------------------------------------------------

col_grandmean_g12 = mean(g12,2);
g12_demean = nan(size(g12));
for i = 1:3
    g12_demean(i,:) = g12(i,:) - mean(g12(i,:));
end

% PCA
[U, S, V] = svd(g12_demean, 'econ');

% eigenvalues
eigvals = diag(S).^2;
eigvals = eigvals/sum(eigvals);

% first component
g12_pca_first_comp = U(:,1)*S(1,1);
g12_pca_first_comp_weights = V(:,1)';

% reconstructed colors
luv_g12_orig = g12_pca_first_comp * g12_pca_first_comp_weights + repmat(col_grandmean_g12, [1, length(g12_pca_first_comp_weights)]);
%luv_g2 = g12_pca_first_comp * g2_pca_first_comp_weights + repmat(col_grandmean_g12, [1, length(g2_pca_first_comp_weights)]);

%this lets you evenly sample the vector, rather than jsut sampling at sites where there are data points:
min_g12 = min(g12_pca_first_comp_weights);
max_g12 = max(g12_pca_first_comp_weights);
g12_vect_points = linspace(min_g12,max_g12,1000);
luv_g12 = g12_pca_first_comp * g12_vect_points + repmat(col_grandmean_g12, [1, length(g12_vect_points)]);


[R, rPval] = corr(b12_pca_first_comp_weights',g12_pca_first_comp_weights'); %get correlation coefficient

luvim_g12 = zeros(size(luv_g12,2),1,3); %initialize the matrix for the best fit line RGB vals
for i = 1:size(luv_g12,2)
    luvim_g12(i,1,2) = luv_g12(1,i);    %function expects L in first dim, u in sepercept, y in third
    luvim_g12(i,1,3) = luv_g12(2,i);   %function expects L in first dim, u in sepercept, y in third
    luvim_g12(i,1,1) = luv_g12(3,i);  %function expects L in first dim, u in sepercept, y in third
end
rgb_g12 = squeeze(Luv2RGB(luvim_g12))';

min_bin_g12 = min(g12_pca_first_comp_weights);
max_bin_g12 = max(g12_pca_first_comp_weights);

%use the more continuous map here:
rgb_g12 = rgb_g12';
Ig12 = ones(10,1000,3);

for i = 1:1000
   
Ig12(:,i,1)=rgb_g12(i,1);
Ig12(:,i,2)=rgb_g12(i,2);
Ig12(:,i,3)=rgb_g12(i,3);

end


figure(2)
hpanel = [7 8 9];
cpanel = [10 11 12];
subplot(4,3,hpanel(pop))
hist(g12_pca_first_comp_weights,100)
xlim([min_bin_g12 max_bin_g12])
xlabel('PC GK')
ylabel('# of color matches')
if dataset ==21
   title('All Subjects')
elseif dataset ==20
    title('Naive Subjects')
elseif dataset ==19
    title('Not-Naive Subjects')
end
%  map = rgb_g12;
%  cmap = colormap(map);
%  cmapg = cmap;
%  colormap(cmapg)
% h = colorbar;
% set(h,'XTickLabel',{})
% set(h,'location','SouthOutside')
subplot(4,3,cpanel(pop))
Ig12s = [Ig12;Ig12;Ig12;Ig12;Ig12;Ig12;Ig12;Ig12];
imshow(Ig12s);
imshow(Ig12s);



figure(1)
if dataset == 21
subplot(2,3,1)
title('All Subjects')
elseif dataset == 20
    subplot(2,3,2)
    title('Naive Subjects')
    elseif dataset == 19
    subplot(2,3,3)
    title('Not-Naive Subjects')
end
x = b12_pca_first_comp_weights;
y = g12_pca_first_comp_weights;
method = 'circles';
radius = sqrt((range(x)/30)^2 + (range(y)/30)^2);  %default: sqrt((range(x)/30)^2 + (range(y)/30)^2);
N = 100; %default:100
n = 5;%default:5
po = 3;
if dataset >1
ms = 6; %marker size, default 4
else 
    %po=2;
    ms = 12;
end
%       0 - no plot
%       1 - plots only colored data points (filtered)
%       2 - plots colored data points and contours (filtered)
%       3 - plots only colored data points (unfiltered)
%       4 - plots colored data points and contours (unfiltered)
%           default is 1

data_out=scatplot_RLS_2(x,y,method,radius,N,n,po,ms);
hold on
plot([min_bin_b12 max_bin_b12],[0 0],'-k')
plot([0 0],[min_bin_g12 max_bin_g12],'-k')
axis equal
xlim([min_bin_b12 max_bin_b12])
ylim([min_bin_g12 max_bin_g12])
xlabel(blabel)
ylabel(glabel)
title('ALL DATA')


data_out_all=data_out;
max(data_out.dd);
num_resps_at_each_loc = (data_out.dd./100)*length(data_out.dd);
max_num_resps_for_color_scale_bar_ALL = max(num_resps_at_each_loc);  %use this to set the upper end (dark red) of the color scale bar to indicate #responses overlapping at that location)
cbh=colorbar('h');
set(cbh,'XTickLabel',{'0',max_num_resps_for_color_scale_bar_ALL},'XTick',[0 1])

corr_pcs_ALL = corr(x',y')  %correlate pcs


%find the top X percent of  distribution :
c_dd_p=[];
figure(100)
data_out=scatplot_RLS_2(x,y,method,radius,N,n,4,1);
conts=linspace(min(data_out.dd),max(data_out.dd),10);
for i = 1:10
    data_out=scatplot_RLS_2_CONTOUR(x,y,method,radius,N,n,4,1,conts(i));
clabel(data_out.c,data_out.hs);
if isempty(data_out.c)
      conts=conts(1:i-1);
    break
else

contour_dd = data_out.dd(data_out.dd >= data_out.c(1,1));
c_dd_p(i) = sum(contour_dd);  %gives the percent of the pop in the single contour
end
end


cont_space=[conts; c_dd_p];
find_top_P = abs(top_P-cont_space(2,:));
[d ind_top_P] = min(find_top_P);

CONT=cont_space(1,ind_top_P);  
data_out=scatplot_RLS_2_CONTOUR(x,y,method,radius,N,n,4,1,CONT);
contour_dd = data_out.dd(data_out.dd >= data_out.c(1,1));
c_dd_percent_ALL = sum(contour_dd) ;
contour_dd = data_out.dd(data_out.dd >= data_out.c(1,1));
c_dd_percent_ALL = sum(contour_dd) ;

find_top_P = abs(top_P-cont_space(2,:));
[d ind_top_P] = min(find_top_P);

CONT=cont_space(1,ind_top_P);  
data_out=scatplot_RLS_2_CONTOUR(x,y,method,radius,N,n,4,1,CONT);
contour_dd = data_out.dd(data_out.dd >= data_out.c(1,1));
c_dd_percent_ALL = sum(contour_dd) ;
contour_dd = data_out.dd(data_out.dd >= data_out.c(1,1));
c_dd_percent_ALL = sum(contour_dd) ;


figure(1)
if dataset == 21
subplot(2,3,1)
title('All Subjects')
elseif dataset == 20
    subplot(2,3,2)
    title('Naive Subjects')
    elseif dataset == 19
    subplot(2,3,3)
    title('Not-Naive Subjects')
end
hold on
scatter(data_out.c(1,:),data_out.c(2,:),'.k')

if dataset == 21
figure(1)
subplot(2,3,4)
x_BK = x(GROUP==1);
y_BK = y(GROUP==1);
po=3;
data_out=scatplot_RLS_2(x_BK,y_BK,method,radius,N,n,po,ms);
hold on
plot([min_bin_b12 max_bin_b12],[0 0],'-k')
plot([0 0],[min_bin_g12 max_bin_g12],'-k')
axis equal
xlim([min_bin_b12 max_bin_b12])
ylim([min_bin_g12 max_bin_g12])
xlabel(blabel)
ylabel(glabel)
title('Blue/Black')

data_out_all=data_out;
max(data_out.dd);
num_resps_at_each_loc = (data_out.dd./100)*length(data_out.dd);
max_num_resps_for_color_scale_bar_BK = max(num_resps_at_each_loc);  %use this to set the upper end (dark red) of the color scale bar to indicate #responses overlapping at that location)
cbh=colorbar('h');
set(cbh,'XTickLabel',{'0',max_num_resps_for_color_scale_bar_BK},'XTick',[0 1]);
corr_pcs_BK = corr(x_BK',y_BK')  %correlate pcs

%find the top X percent of each cluster :
c_dd_p=[];
figure(100);
data_out=scatplot_RLS_2(x_BK,y_BK,method,radius,N,n,4,1);
conts=linspace(min(data_out.dd),max(data_out.dd),10);
for i = 1:10

    data_out=scatplot_RLS_2_CONTOUR(x_BK,y_BK,method,radius,N,n,4,1,conts(i));
clabel(data_out.c,data_out.hs);
if isempty(data_out.c)
      conts=conts(1:i-1);
    break
else

contour_dd = data_out.dd(data_out.dd >= data_out.c(1,1));
c_dd_p(i) = sum(contour_dd);  %gives the percent of the pop in the single contour
end
end
cont_space=[conts; c_dd_p];
find_top_P = abs(top_P-cont_space(2,:));
[d ind_top_P] = min(find_top_P);

CONT=cont_space(1,ind_top_P);  
data_out=scatplot_RLS_2_CONTOUR(x_BK,y_BK,method,radius,N,n,4,1,CONT);
contour_dd = data_out.dd(data_out.dd >= data_out.c(1,1));
c_dd_percent_BK = sum(contour_dd) ;
contour_dd = data_out.dd(data_out.dd >= data_out.c(1,1));
c_dd_percent_BK = sum(contour_dd) ;

find_top_P = abs(top_P-cont_space(2,:));
[d ind_top_P] = min(find_top_P);

CONT=cont_space(1,ind_top_P);  
data_out=scatplot_RLS_2_CONTOUR(x_BK,y_BK,method,radius,N,n,4,1,CONT);
contour_dd = data_out.dd(data_out.dd >= data_out.c(1,1));
c_dd_percent_BK = sum(contour_dd) ;
contour_dd = data_out.dd(data_out.dd >= data_out.c(1,1));
c_dd_percent_BK = sum(contour_dd) ;




figure(1)
subplot(2,3,4)
hold on
scatter(data_out.c(1,:),data_out.c(2,:),'.k')



figure(1)
subplot(2,3,5)
%subplot(1,5,3)
x_WG = x(GROUP==2);
y_WG = y(GROUP==2);
po=3;
data_out=scatplot_RLS_2(x_WG,y_WG,method,radius,N,n,po,ms);
hold on
plot([min_bin_b12 max_bin_b12],[0 0],'-k')
plot([0 0],[min_bin_g12 max_bin_g12],'-k')
axis equal
xlim([min_bin_b12 max_bin_b12])
ylim([min_bin_g12 max_bin_g12])
xlabel(blabel)
ylabel(glabel)
title('White/Gold')

data_out_all=data_out;
max(data_out.dd);
num_resps_at_each_loc = (data_out.dd./100)*length(data_out.dd);
max_num_resps_for_color_scale_bar_WG = max(num_resps_at_each_loc) ; %use this to set the upper end (dark red) of the color scale bar to indicate #responses overlapping at that location)
cbh=colorbar('h');
set(cbh,'XTickLabel',{'0',max_num_resps_for_color_scale_bar_WG},'XTick',[0 1]);
corr_pcs_WG = corr(x_WG',y_WG')  %correlate pcs


c_dd_p=[];
figure(100);
data_out=scatplot_RLS_2(x_WG,y_WG,method,radius,N,n,4,1);
conts=linspace(min(data_out.dd),max(data_out.dd),10);
for i = 1:10

    data_out=scatplot_RLS_2_CONTOUR(x_WG,y_WG,method,radius,N,n,4,1,conts(i));
clabel(data_out.c,data_out.hs);
if isempty(data_out.c)
      conts=conts(1:i-1);
    break
else

contour_dd = data_out.dd(data_out.dd >= data_out.c(1,1));
c_dd_p(i) = sum(contour_dd);  %gives the percent of the pop in the single contour
end
end
cont_space=[conts; c_dd_p];
find_top_P = abs(top_P-cont_space(2,:));
[d ind_top_P] = min(find_top_P);

CONT=cont_space(1,ind_top_P);  
data_out=scatplot_RLS_2_CONTOUR(x_WG,y_WG,method,radius,N,n,4,1,CONT);

contour_dd = data_out.dd(data_out.dd >= data_out.c(1,1));
c_dd_percent = sum(contour_dd) ;
contour_dd = data_out.dd(data_out.dd >= data_out.c(1,1));
c_dd_percent_wg = sum(contour_dd) ;

find_top_P = abs(top_P-cont_space(2,:));
[d ind_top_P] = min(find_top_P);

CONT=cont_space(1,ind_top_P);  
data_out=scatplot_RLS_2_CONTOUR(x_WG,y_WG,method,radius,N,n,4,1,CONT);
contour_dd = data_out.dd(data_out.dd >= data_out.c(1,1));
c_dd_percent_BK = sum(contour_dd) ;
contour_dd = data_out.dd(data_out.dd >= data_out.c(1,1));
c_dd_percent_WG = sum(contour_dd) ;


figure(1)
subplot(2,3,5)
hold on
scatter(data_out.c(1,:),data_out.c(2,:),'.k')


figure(1)
subplot(2,3,6)
%subplot(1,5,4)
x_BG = x(GROUP==3);
y_BG = y(GROUP==3);
po=3;
data_out=scatplot_RLS_2(x_BG,y_BG,method,radius,N,n,po,ms);
hold on
plot([min_bin_b12 max_bin_b12],[0 0],'-k')
plot([0 0],[min_bin_g12 max_bin_g12],'-k')
axis equal
xlim([min_bin_b12 max_bin_b12])
ylim([min_bin_g12 max_bin_g12])
xlabel(blabel)
ylabel(glabel)
title('Blue/Brown')

data_out_all=data_out;
max(data_out.dd);
num_resps_at_each_loc = (data_out.dd./100)*length(data_out.dd);
max_num_resps_for_color_scale_bar_BG = max(num_resps_at_each_loc) ; %use this to set the upper end (dark red) of the color scale bar to indicate #responses overlapping at that location)
cbh=colorbar('h');
set(cbh,'XTickLabel',{'0',max_num_resps_for_color_scale_bar_BG},'XTick',[0 1]);
corr_pcs_BG = corr(x_BG',y_BG')  %correlate pcs


c_dd_p=[];
figure(100);
data_out=scatplot_RLS_2(x_BG,y_BG,method,radius,N,n,4,1);
conts=linspace(min(data_out.dd),max(data_out.dd),10);
for i = 1:10

    data_out=scatplot_RLS_2_CONTOUR(x_BG,y_BG,method,radius,N,n,4,1,conts(i));
clabel(data_out.c,data_out.hs);
if isempty(data_out.c)
      conts=conts(1:i-1);
    break
else

contour_dd = data_out.dd(data_out.dd >= data_out.c(1,1));
c_dd_p(i) = sum(contour_dd);  %gives the percent of the pop in the single contour
end
end
cont_space=[conts; c_dd_p];
find_top_P = abs(top_P-cont_space(2,:));
[d ind_top_P] = min(find_top_P);

CONT=cont_space(1,ind_top_P);  %examine plot 799 and results inside cont_space and set this appropriately
data_out=scatplot_RLS_2_CONTOUR(x_BG,y_BG,method,radius,N,n,4,1,CONT);
contour_dd = data_out.dd(data_out.dd >= data_out.c(1,1));
c_dd_percent = sum(contour_dd); 

figure(1)
subplot(2,3,6)
hold on
scatter(data_out.c(1,:),data_out.c(2,:),'.k')
set(gcf,'color','w');
set(gcf, 'Position', [229, 194, 1325, 727])

close Figure 100

end

end


I=imread('dress.tif');
figure(3)
imshow(I)

