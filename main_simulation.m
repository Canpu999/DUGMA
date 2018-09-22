%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  These examples corresponds to simulation part in our paper and supplementary materials

clc
clear all
close all
warning('off','all')
global A


% Choose 2D registration or 3D registration
D=2;                       % D=2 means 2D registration, D=3 means 3D registration


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set the random range for different simulated factors
if D==2
    shape=round(100*rand); if (shape==0) shape=1; end    % choose one shape
    rand_degree=30;    % random range for rotation angle [-rand_degree, rand_degree]
    rand_outliers=50;   % random range for outliers [0, rand_outliers]
    rand_noise=0.1;      % random range for noise [0, rand_noise]
    rand_miss_rate=0.1; % random range for occlusion [0, rand_miss_rate]
    St_data=[pwd,'/dataset_all/2D_SHAPE_MODEL/'];    % 100 different 2D shapes
    %St_data=[pwd,'/dataset_all/2D_Fish_MODEL/dataset2D_500/'];        % 100 different 2D fish
    %St_data=[pwd,'/dataset_all/2D_Fish_MODEL/dataset2D_400/'];        % 100 different 2D fish
    %St_data=[pwd,'/dataset_all/2D_Fish_MODEL/dataset2D_300/'];        % 100 different 2D fish
    %St_data=[pwd,'/dataset_all/2D_Fish_MODEL/dataset2D_200/'];        % 100 different 2D fish
    %St_data=[pwd,'/dataset_all/2D_Fish_MODEL/dataset2D_100/'];        % 100 different 2D fish
else
    shape=1;         %Because regulation of Trimbot, there is only 1 downsampled sparse model released
    rand_degree=30;   % random range for rotation angle [-rand_degree, rand_degree] around each axis
    rand_outliers=100; % random range for outliers [0, rand_outliers]
    rand_noise=0.1;   % random range for noise [0, rand_noise]
    rand_miss_rate=0.1; % random range for occlusion [0, rand_miss_rate]
    St_data=[pwd,'/dataset_all/3D_SHAPE/'];
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% produce synthetic point cloud X, Y and their corresponding covariance
% X_C, Y_C
[X,X_C, Y, Y_C, R_G, t_G]=produce_point_cloud(rand_degree, rand_noise, rand_outliers, rand_miss_rate, shape, St_data, D);


 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % set parameter for DUGMA
            Max_Iteration=700;
            accuracy_rotation=0.0001;
            accuracy_translation=0.00000001;
            accuracy_sig=0.000000001;
            R0=eye(D); t0=zeros(D,1);
        
     
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % Register two point cloud
    [R,t]=DUGMA(X,Y,X_C,Y_C,R0,t0,Max_Iteration, accuracy_rotation, accuracy_translation, accuracy_sig)

    
    
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Error
    D=size(X,1);
    error_R=sqrt(sum(sum((eye(D)-R*pinv(R_G)).^2)))
    error_t=sqrt(sum((t-t_G).^2))





