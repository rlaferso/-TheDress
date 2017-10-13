%%%This script generates the components of Figure 2 from the publication: 
% Lafer-Sousa, R., & Conway, B. R. (2017). #thedress: Categorical perception of an ambiguous color image. Journal of Vision
% Figure legend:
% Figure 2. K-means clustering of color-matching data favor a two- or three-component model over a single-component distribution.
% Plots summarize the results from k-means clustering assessment (via the gap method; Tibshirani et al., 2001) of the color-matching
% data presented in Figure 1A (N = 2,200 subjects). (A) The gap statistic computed as a function of the number of clusters, for colormatching
% data (principal-component analysis weights) obtained for the upper regions of the dress, using all the data from the online
% population. Dashed line indicates the optimal k clusters. (B) Bar plot showing the cluster assignments of the color matches, binned by
% the color terms used by the subjects to describe the dress. (C) RGB values of the color matches, sorted by cluster assignment from
% (B); each thin horizontal bar shows the color matches for a single subject. (D?F) As for (A?C), but for color matches made to the
% bottom regions of the dress. The gap analysis compares the within-cluster dispersion of a k-component model to its expectation
% under a null model of a single component. The algorithm seeks to identify the smallest number of clusters satisfying Gap(k) 
% GAPMAX  SE(GAPMAX), where k is the number of clusters, Gap(k) is the gap value for the clustering solution with k clusters,
% GAPMAX is the largest gap value, and SE(GAPMAX) is the standard error corresponding to the largest gap value. The optimal k
% solution for the distribution of upper dress-region color matches is two clusters, and for the lower dress regions it is three, confirming
% the suspected nonunimodality of the underlying population distribution.


%%%The script was written by Rosa Lafer-Sousa. (for questions, contact: rlaferso@mit.edu)
    % Note: this code was modified from its original format to produce only the contents of Figure 1 from the publication. last
    % modified: 10/13/2017.  Matlab version 2016B (originally written in with Matlab 2015B)

%%%Notes on running the code:
% The code calls on support scripts, provided in the present directory:
        %barwitherr.m
        %Luv2RGB.m
        %scatplot_RLS_2.m
        %scatplot_RLS_CONTOUR.m
% The code loads data from several .mat files that were generated from the raw data provided in the folder RAW DATA:
    %All the data we collected online (Amazon's Mechanical Turk) from the 'SCALE Experiment' and 'MAIN Experiment' (all scales presented); N = 2200:
        %  TURK_MAIN_and_SCALE1036100150_BK_ALL.mat   'Blue/Black' reporters
        %  TURK_MAIN_and_SCALE1036100150_WG_ALL.mat   'White/Gold' reporters
        %  TURK_MAIN_and_SCALE1036100150_BG_ALL.mat   'Blue/Brown' reporters
        %  TURK_MAIN_and_SCALE1036100150_OT_ALL.mat   'Other color' reporters
   
% Running this script will generate 1 figure file, containing all the
% panels of the published figure.
 

   
clear all; close all; clc

for regions_up_low = 1:2
    numclusts=[2 3];
test_3=numclusts(regions_up_low); %set to 2 to test 2 clusters; 3 for 3 clusts; 1 for 1 clust
distance_method = 'cityblock'; %'sqEuclidean'; %'cityblock';  
%regions_up_low = 1; %1 for upper dress regions, 2 for lower; 3 for both
make_tapestry = 1; rand_shuffle = 2;

      filenames =   {'TURK_MAIN_and_SCALE1036100150_BK_ALL'...
       'TURK_MAIN_and_SCALE1036100150_WG_ALL'...
        'TURK_MAIN_and_SCALE1036100150_BG_ALL'...
        'TURK_MAIN_and_SCALE1036100150_OT_ALL'};%ALL subjects from the SCALE expt and MAIN turk (all scales presented) N = 
 
    


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

min_bin_b12 = min(b12_pca_first_comp_weights)
max_bin_b12 = max(b12_pca_first_comp_weights)



rgb_b12 = rgb_b12';
Ib12 = ones(10,1000,3);

for i = 1:1000
   
Ib12(:,i,1)=rgb_b12(i,1);
Ib12(:,i,2)=rgb_b12(i,2);
Ib12(:,i,3)=rgb_b12(i,3);

end



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


[R, rPval] = corr(b12_pca_first_comp_weights',g12_pca_first_comp_weights') %get correlation coefficient

luvim_g12 = zeros(size(luv_g12,2),1,3); %initialize the matrix for the best fit line RGB vals
for i = 1:size(luv_g12,2)
    luvim_g12(i,1,2) = luv_g12(1,i);    %function expects L in first dim, u in sepercept, y in third
    luvim_g12(i,1,3) = luv_g12(2,i);   %function expects L in first dim, u in sepercept, y in third
    luvim_g12(i,1,1) = luv_g12(3,i);  %function expects L in first dim, u in sepercept, y in third
end
rgb_g12 = squeeze(Luv2RGB(luvim_g12))';

min_bin_g12 = min(g12_pca_first_comp_weights)
max_bin_g12 = max(g12_pca_first_comp_weights)


%use the more continuous map here:
rgb_g12 = rgb_g12';
Ig12 = ones(10,1000,3);

for i = 1:1000
   
Ig12(:,i,1)=rgb_g12(i,1);
Ig12(:,i,2)=rgb_g12(i,2);
Ig12(:,i,3)=rgb_g12(i,3);

end



%%%RUN CLUSTERING ANALYSIS:

X = [b12_pca_first_comp_weights' g12_pca_first_comp_weights'];
maxClust = 13;

%X=X';

%-- evaluate kmeans solutions...
% eva = evalclusters(X,'kmeans','CalinskiHarabasz','KList',[1:13])
% % figure(900)
% % xlabel('number of clusters')
% % ylabel('Calinski-Harabasz value')
% % subplot(1,3,2)
% % CH = plot(eva);
% % ax = gca;ax.Box='off';ax.LineWidth = 1.5;ax.FontSize = 16;
% %     set(CH,{'Color'},{[0 0 0]});  
% %     set(CH,{'LineWidth'},{1.5}); 
% % set(900,'color',[1 1 1])
% 
% 
% K_CH = eva.OptimalK  %Calinski-Harabasz test

% eva = evalclusters(X,'kmeans','Silhouette','KList',[1:13],'Distance',distance_method)
% % figure(900)
% % subplot(1,3,3)
% % xlabel('number of clusters')
% % ylabel('Sil')
% % SI = plot(eva);
% % ax = gca;ax.Box='off';ax.LineWidth = 1.5;ax.FontSize = 16;
% %     set(SI,{'Color'},{[0 0 0]});  
% %     set(SI,{'LineWidth'},{1.5}); 
% % set(900,'color',[1 1 1])
% 
% K_Sil = eva.OptimalK %Silhouette test

eva = evalclusters(X,'kmeans','Gap','KList',[1:13],'Distance' , distance_method )
figure(1)
Gpanel=[1 4];Gmin=[.64 .8];Gmax=[.85 1];
subplot(2,3,Gpanel(regions_up_low))
plot(eva)
hold on 
stem(eva.OptimalK,eva.CriterionValues(eva.OptimalK),'r')
ylim([Gmin(regions_up_low) Gmax(regions_up_low)]);
xlabel('number of clusters')
ylabel('Gap')
  %   'YTick', [1 2 3], 'Yticklabel',{'B/K', 'W/G', 'B/B'} , 'box', 'off','fontsize', 12);
  set(gca, 'box', 'off')

K_Gap = eva.OptimalK


opts = statset('Display','final');
 [idx,C,sumd,dists] = kmeans(X,2,'Distance',distance_method ,...
    'Replicates',5,'Options',opts);
clusters_avg_internal_dist = mean(dists,1)



if test_3==2;

opts = statset('Display','final');
 [idx,C,sumd,dists] = kmeans(X,2,'Distance',distance_method ,...
    'Replicates',5,'Options',opts);
clusters_avg_internal_dist = mean(dists,1)


%find the percent of BK, WG, and BG subjects that fall in clust 1 and clust 2.
idx_clust1 = (idx==1);
idx_clust2 = (idx==2);
bk = (GROUP==1);
wg = (GROUP==2);
bg = (GROUP==3);
ot = (GROUP==4);

bk_clust1_match = bk+idx_clust1;
num_bk_clust1_match = sum(bk_clust1_match==2);
pct_bk_clust1_match = sum(bk_clust1_match==2)/sum(bk)*100
wg_clust1_match = wg+idx_clust1;
num_wg_clust1_match = sum(wg_clust1_match==2);
pct_wg_clust1_match = sum(wg_clust1_match==2)/sum(wg)*100
bg_clust1_match = bg+idx_clust1;
num_bg_clust1_match = sum(bg_clust1_match==2);
pct_bg_clust1_match = sum(bg_clust1_match==2)/sum(bg)*100
ot_clust1_match = ot+idx_clust1;
num_ot_clust1_match = sum(ot_clust1_match==2);
pct_ot_clust1_match = sum(ot_clust1_match==2)/sum(ot)*100

bk_clust2_match = bk+idx_clust2;
num_bk_clust2_match = sum(bk_clust2_match==2);
pct_bk_clust2_match = sum(bk_clust2_match==2)/sum(bk)*100
wg_clust2_match = wg+idx_clust2;
num_wg_clust2_match = sum(wg_clust2_match==2);
pct_wg_clust2_match = sum(wg_clust2_match==2)/sum(wg)*100
bg_clust2_match = bg+idx_clust2;
num_bg_clust2_match = sum(bg_clust2_match==2);
pct_bg_clust2_match = sum(bg_clust2_match==2)/sum(bg)*100
ot_clust2_match = ot+idx_clust2;
num_ot_clust2_match = sum(ot_clust2_match==2);
pct_ot_clust2_match = sum(ot_clust2_match==2)/sum(ot)*100

   
   figure(1)
   subplot(2,3,2)
   clust_1 = [num_bk_clust1_match num_wg_clust1_match num_bg_clust1_match num_ot_clust1_match];
clust_2 = [num_bk_clust2_match num_wg_clust2_match num_bg_clust2_match num_ot_clust2_match];
if num_wg_clust1_match > num_bk_clust1_match
clusters12 = [clust_1; clust_2];
renum_clust_tap = 0;
else
    clusters12 = [clust_2; clust_1];
    renum_clust_tap = 1;
end
bar(clusters12)
xlim([.5 3.5])
ylabel('# of subjects assigned to cluster')
legend('BK','WG','BB','OT',...
       'Location','NE')
   set(gca, 'XTickLabel',{'Cluster 1','Cluster 2'}, 'box', 'off')
   colormap bone
   
   set(gcf,'color','w');
    
  
if pct_bk_clust1_match > pct_bk_clust2_match
    x_bkc=X(bk_clust1_match==2,1);
    y_bkc=X(bk_clust1_match==2,2);
    x_wgc=X(wg_clust2_match==2,1);
    y_wgc=X(wg_clust2_match==2,2);

else
    x_bkc=X(bk_clust2_match==2,1);
    y_bkc=X(bk_clust2_match==2,2);
    x_wgc=X(wg_clust1_match==2,1);
    y_wgc=X(wg_clust1_match==2,2);
end

x_clust1 = X(idx==1,1);
y_clust1 = X(idx==1,2);
x_clust2 = X(idx==2,1);
y_clust2 = X(idx==2,2);




if make_tapestry ==1
%make tapestry:
%try to make tapestry...
RGB_A=RGB_A/255;
RGB_B=RGB_B/255;
RGB_C=RGB_C/255;
RGB_D=RGB_D/255; %rgb vals of all samples region D (iv)
% 


rgbA_clust1 = RGB_A(idx==1,:);
rgbA_clust2 = RGB_A(idx==2,:);
rgbB_clust1 = RGB_B(idx==1,:);
rgbB_clust2 = RGB_B(idx==2,:);
rgbC_clust1 = RGB_C(idx==1,:);
rgbC_clust2 = RGB_C(idx==2,:);
rgbD_clust1 = RGB_D(idx==1,:);
rgbD_clust2 = RGB_D(idx==2,:);


pixel_r= 2 ;pixel_c=50;

for tap = 1:2
    
if tap ==1
    rgbA_cluster = rgbA_clust1; rgbB_cluster = rgbB_clust1;rgbC_cluster = rgbC_clust1;rgbD_cluster = rgbD_clust1;
elseif tap ==2 
    rgbA_cluster = rgbA_clust2; rgbB_cluster = rgbB_clust2;rgbC_cluster = rgbC_clust2;rgbD_cluster = rgbD_clust2;
end


clust_tap=[];

% dists_2sort = avgclust_dist(idx==tap);
% [Y,I] = sort(dists_2sort,1,'descend');
% [Y2,I2] = sortrows(dists(idx==tap,:),[-1 -2]);


num_subjs = size(rgbA_cluster,1);
if rand_shuffle == 2
    p = randperm(num_subjs,num_subjs);
elseif rand_shuffle == 1 || rand_shuffle == 3
     p = 1:num_subjs;
elseif rand_shuffle == 4
     p = I;
elseif rand_shuffle == 5
     p = I2;
end
for i = 1:size(rgbA_cluster,1)
    a = ones(pixel_r,pixel_c,3);
    a(:,:,1) = rgbA_cluster(p(i),1);
    a(:,:,2) = rgbA_cluster(p(i),2);
    a(:,:,3) = rgbA_cluster(p(i),3);
    
    b = ones(pixel_r,pixel_c,3);
    b(:,:,1) = rgbB_cluster(p(i),1);
    b(:,:,2) = rgbB_cluster(p(i),2);
    b(:,:,3) = rgbB_cluster(p(i),3);
    
    c = ones(pixel_r,pixel_c,3);
    c(:,:,1) = rgbC_cluster(p(i),1);
    c(:,:,2) = rgbC_cluster(p(i),2);
    c(:,:,3) = rgbC_cluster(p(i),3);
    
    d = ones(pixel_r,pixel_c,3);
    d(:,:,1) = rgbD_cluster(p(i),1);
    d(:,:,2) = rgbD_cluster(p(i),2);
    d(:,:,3) = rgbD_cluster(p(i),3);
    clust_tap = [clust_tap; a,b,d,c];
end
if tap ==1
    clust_1_tap = clust_tap;
elseif tap ==2
    clust_2_tap = clust_tap;

end
end


maxg=max([size(clust_1_tap,1) size(clust_2_tap,1)]);
if size(clust_1_tap,1) < maxg
    clust_1_tap = [ones(maxg-size(clust_1_tap,1),200,3); clust_1_tap];
elseif size(clust_2_tap,1) < maxg
    clust_2_tap = [ones(maxg-size(clust_2_tap,1),200,3); clust_2_tap];
end
white_strip = ones(maxg, 200, 3);

if renum_clust_tap ==0
im_tap=[clust_1_tap white_strip clust_2_tap white_strip white_strip];
elseif renum_clust_tap==1
    im_tap=[clust_1_tap white_strip  clust_2_tap white_strip white_strip];
end

figure(1)
subplot(2,3,3)
imshow(im_tap)


end
 
end



if test_3 == 3
%------ 3 clusters...---------------------------

[idx,C,sumd,dists] = kmeans(X,3,'Distance',distance_method ,...
   'Replicates',1,'Options',opts);


%find the percent of BK, WG, and BG subjects that fall in clust 1,2, and 3 
idx_clust1 = (idx==1);
idx_clust2 = (idx==2);
idx_clust3 = (idx==3);
bk = (GROUP==1);
wg = (GROUP==2);
bg = (GROUP==3);
ot = (GROUP==4);

bk_clust1_match = bk+idx_clust1;
num_bk_clust1_match = sum(bk_clust1_match==2);
pct_bk_clust1_match = sum(bk_clust1_match==2)/sum(bk)*100
wg_clust1_match = wg+idx_clust1;
num_wg_clust1_match = sum(wg_clust1_match==2);
pct_wg_clust1_match = sum(wg_clust1_match==2)/sum(wg)*100
bg_clust1_match = bg+idx_clust1;
num_bg_clust1_match = sum(bg_clust1_match==2);
pct_bg_clust1_match = sum(bg_clust1_match==2)/sum(bg)*100
ot_clust1_match = ot+idx_clust1;
num_ot_clust1_match = sum(ot_clust1_match==2);
pct_ot_clust1_match = sum(ot_clust1_match==2)/sum(ot)*100

bk_clust2_match = bk+idx_clust2;
num_bk_clust2_match = sum(bk_clust2_match==2);
pct_bk_clust2_match = sum(bk_clust2_match==2)/sum(bk)*100
wg_clust2_match = wg+idx_clust2;
num_wg_clust2_match = sum(wg_clust2_match==2);
pct_wg_clust2_match = sum(wg_clust2_match==2)/sum(wg)*100
bg_clust2_match = bg+idx_clust2;
num_bg_clust2_match = sum(bg_clust2_match==2);
pct_bg_clust2_match = sum(bg_clust2_match==2)/sum(bg)*100
ot_clust2_match = ot+idx_clust2;
num_ot_clust2_match = sum(ot_clust2_match==2);
pct_ot_clust2_match = sum(ot_clust2_match==2)/sum(ot)*100

bk_clust3_match = bk+idx_clust3;
num_bk_clust3_match = sum(bk_clust3_match==2);
pct_bk_clust3_match = sum(bk_clust3_match==2)/sum(bk)*100
wg_clust3_match = wg+idx_clust3;
num_wg_clust3_match = sum(wg_clust3_match==2);
pct_wg_clust3_match = sum(wg_clust3_match==2)/sum(wg)*100
bg_clust3_match = bg+idx_clust3;
num_bg_clust3_match = sum(bg_clust3_match==2);
pct_bg_clust3_match = sum(bg_clust3_match==2)/sum(bg)*100
ot_clust3_match = ot+idx_clust3;
num_ot_clust3_match = sum(ot_clust3_match==2);
pct_ot_clust3_match = sum(ot_clust3_match==2)/sum(ot)*100


clust_1 = [num_bk_clust1_match num_wg_clust1_match num_bg_clust1_match num_ot_clust1_match];
clust_2 = [num_bk_clust2_match num_wg_clust2_match num_bg_clust2_match num_ot_clust2_match];
clust_3 = [num_bk_clust3_match num_wg_clust3_match num_bg_clust3_match num_ot_clust3_match];


if num_wg_clust1_match > num_bk_clust1_match
clusters123 = [clust_1; clust_2; clust_3;];
renum_clust_tap = 0;
else
    clusters123 = [clust_3; clust_2; clust_1;];
    renum_clust_tap = 1;
end


figure(1)
subplot(2,3,5)
bar(clusters123)
ylabel('# of subjects assigned to cluster')
xlim([.5 3.5])
% legend('BK','WG','BB','OT',...
%        'Location','NE')
   set(gca, 'XTickLabel',{'Cluster 1','Cluster 2','Cluster 3'}, 'box', 'off')
   colormap bone
   set(gcf,'color','w');
 
 
if make_tapestry ==1
%make tapestry:
%try to make tapestry...
RGB_A=RGB_A/255;
RGB_B=RGB_B/255;
RGB_C=RGB_C/255;
RGB_D=RGB_D/255; %rgb vals of all samples region D (iv)
% 



rgbA_clust1 = RGB_A(idx==1,:);
rgbA_clust2 = RGB_A(idx==2,:);
rgbA_clust3 = RGB_A(idx==3,:);
rgbB_clust1 = RGB_B(idx==1,:);
rgbB_clust2 = RGB_B(idx==2,:);
rgbB_clust3 = RGB_B(idx==3,:);
rgbC_clust1 = RGB_C(idx==1,:);
rgbC_clust2 = RGB_C(idx==2,:);
rgbC_clust3 = RGB_C(idx==3,:);
rgbD_clust1 = RGB_D(idx==1,:);
rgbD_clust2 = RGB_D(idx==2,:);
rgbD_clust3 = RGB_D(idx==3,:);


pixel_r= 2 ;pixel_c=50;

for tap = 1:3
    
if tap ==1
    rgbA_cluster = rgbA_clust1; rgbB_cluster = rgbB_clust1;rgbC_cluster = rgbC_clust1;rgbD_cluster = rgbD_clust1;
elseif tap ==2 
    rgbA_cluster = rgbA_clust2; rgbB_cluster = rgbB_clust2;rgbC_cluster = rgbC_clust2;rgbD_cluster = rgbD_clust2;
elseif tap ==3 
    rgbA_cluster = rgbA_clust3; rgbB_cluster = rgbB_clust3;rgbC_cluster = rgbC_clust3;rgbD_cluster = rgbD_clust3;
end

clust_tap=[];



num_subjs = size(rgbA_cluster,1);
if rand_shuffle == 2
    p = randperm(num_subjs,num_subjs);
elseif rand_shuffle == 1 || rand_shuffle == 3
     p = 1:num_subjs;

end

for i = 1:size(rgbA_cluster,1)
    
    a = ones(pixel_r,pixel_c,3);
    a(:,:,1) = rgbA_cluster(p(i),1);
    a(:,:,2) = rgbA_cluster(p(i),2);
    a(:,:,3) = rgbA_cluster(p(i),3);
    
    b = ones(pixel_r,pixel_c,3);
    b(:,:,1) = rgbB_cluster(p(i),1);
    b(:,:,2) = rgbB_cluster(p(i),2);
    b(:,:,3) = rgbB_cluster(p(i),3);
    
    c = ones(pixel_r,pixel_c,3);
    c(:,:,1) = rgbC_cluster(p(i),1);
    c(:,:,2) = rgbC_cluster(p(i),2);
    c(:,:,3) = rgbC_cluster(p(i),3);
    
    d = ones(pixel_r,pixel_c,3);
    d(:,:,1) = rgbD_cluster(p(i),1);
    d(:,:,2) = rgbD_cluster(p(i),2);
    d(:,:,3) = rgbD_cluster(p(i),3);
    clust_tap = [clust_tap; a,b,d,c];
    %clust_tap=clust_tap*255;
end
if tap ==1
    clust_1_tap = clust_tap;
elseif tap ==2
    clust_2_tap = clust_tap;
elseif tap ==3
    clust_3_tap = clust_tap;
end
end


maxg=max([size(clust_1_tap,1) size(clust_2_tap,1) size(clust_3_tap,1)]);
if size(clust_1_tap,1) < maxg
    clust_1_tap = [ones(maxg-size(clust_1_tap,1),200,3); clust_1_tap];
end
if size(clust_2_tap,1) < maxg
    clust_2_tap = [ones(maxg-size(clust_2_tap,1),200,3); clust_2_tap];
end
if   size(clust_3_tap,1) < maxg
    clust_3_tap = [ones(maxg-size(clust_3_tap,1),200,3); clust_3_tap];
end
white_strip = ones(maxg, 200, 3);
if renum_clust_tap ==0
im_tap=[clust_1_tap white_strip clust_2_tap white_strip clust_3_tap];
elseif renum_clust_tap ==1
im_tap=[clust_3_tap white_strip clust_2_tap white_strip clust_1_tap];
end

figure(1)
subplot(2,3,6)
imshow(im_tap)

set(gcf,'color','w');
set(gcf, 'Position', [1, 1, 1111, 570])


 end

 
 
end 
end
