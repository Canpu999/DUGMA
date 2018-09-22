%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This corresponds to the Sec. 4.2 in our paper


clc
clear all
close all


 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         % Example:  Real Kinect Application setting in Our Paper 
         % You could balance these parameters by yourself to speed it up
         
            Max_Iteration=300;
            accuracy_rotation=0.0001;
            accuracy_translation=0.00000001;
            accuracy_sig=0.000000001;
            
            k=1;    % The k scene from 2 kinects, k could be 1~30
            load([pwd,'/dataset_all/Kinect_Application/',num2str(k),'.mat']); 
            R0=eye(3); t0=zeros(3,1);
            % figure 1 in our paper
            %load([pwd,'/dataset_all/Kinect_Application/',num2str(20),'.mat']);  R0=eye(3); t0=zeros(3,1);        
     
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    % Register two point cloud
    [R,t]=DUGMA(X,Y,X_C,Y_C,R0,t0,Max_Iteration, accuracy_rotation, accuracy_translation, accuracy_sig)

    % Error
    D=size(X,1);
    error_R=sqrt(sum(sum((eye(D)-R*pinv(R_G)).^2)))
    error_t=sqrt(sum((t-t_G).^2))









