function [X,Y,X_C,Y_C]=noise_and_sampling3D(Model,R_G,t_G,Noi,miss_rate)
   % (noise,1) (degree,2) (outliers,3) (fa,4)
   %  Noi is the noise level added to the Model
   
   fa=3.4657;

   M=size(Model,2);
   
   occ_point1=round(rand*M);
   if occ_point1==0
       occ_point1=1;
   end   
   dist1=sum(((Model-Model(:,occ_point1))).^2);
   [dist1,index_occ1]=sort(dist1);   
   index_occ1=index_occ1(1:round(miss_rate*M));
   
   occ_point2=round(rand*M);
   if occ_point2==0
       occ_point2=1;
   end   
   dist2=sum(((Model-Model(:,occ_point2))).^2);
   [dist2,index_occ2]=sort(dist2);   
   index_occ2=index_occ2(1:round(miss_rate*M));   
   
   % the sample rate for X and Y should be the same
%    if M>1000
%           sample_rateX=950/M;
%           sample_rateY=920/M;
%    else
%           sample_rateX=0.9;
%           sample_rateY=0.85;
%    end
           sample_rateX=0.9;
           sample_rateY=0.85;   
   
   
   % get the absolute standard deviation on x and y for Model
    temp=Model;
    % calculate the mean
    normal.model=mean(temp');
    normal.model=normal.model';
    %shift 
    temp=temp-repmat(normal.model,1,M);
    % calculate the radius, a little like the standard deviation
    radius=sqrt(sum(sum(temp.^2,2))/M);
   
   
   % absolute standard deviation on x and y for each point 
   ration_x= abs(Noi*randn(1,M));
   ration_y= abs(Noi*randn(1,M));
   ration_z= abs(Noi*randn(1,M));
   
   %fa=5;
   % covariance
   v_x=exp(ration_x*fa);
   v_y=exp(ration_y*fa);
   v_z=exp(ration_z*fa);
  
   % produce noise for X point cloud
   X=[];
   X_C=[];

   noise_x=ration_x.*randn(1,M);
   noise_y=ration_y.*randn(1,M);
   noise_z=ration_z.*randn(1,M);  
   
   Model_X=zeros(3,M);
   Model_X(1,:)=Model(1,:)+noise_x*radius;
   Model_X(2,:)=Model(2,:)+noise_y*radius;
   Model_X(3,:)=Model(3,:)+noise_z*radius;
   

   for i=1:M
       judge=isempty(find(i==index_occ1));
       if (rand<=sample_rateX)  && (judge)
           temp_x=Model_X(:,i);
           temp_x_c=zeros(3,3);
           temp_x_c(1)=v_x(i);
           temp_x_c(5)=v_y(i);
           temp_x_c(9)=v_z(i);
           X=[X temp_x];
           X_C=[X_C temp_x_c];
       end
   end
   
   
   % produce noise for Y point cloud
   Y=[];
   Y_C=[];
   noise_x=ration_x.*randn(1,M);
   noise_y=ration_y.*randn(1,M);
   noise_z=ration_z.*randn(1,M);   
   Model_Y=zeros(3,M);
   Model_Y(1,:)=Model(1,:)+noise_x*radius;
   Model_Y(2,:)=Model(2,:)+noise_y*radius;
   Model_Y(3,:)=Model(3,:)+noise_z*radius;   
   for i=1:M
       judge=isempty(find(i==index_occ2));       
       if (rand<=sample_rateY) && (judge)
           temp_y=Model_Y(:,i);
           temp_y=R_G*temp_y+t_G;
           temp_y_c=zeros(3,3);
           temp_y_c(1)=v_x(i);
           temp_y_c(5)=v_y(i);
           temp_y_c(9)=v_z(i);           
           temp_y_c=R_G*temp_y_c*R_G';
           Y=[Y temp_y];
           Y_C=[Y_C temp_y_c];
       end
   end
   
   
   
  

end