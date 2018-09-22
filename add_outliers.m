function [X]=add_outliers(X,sd,num,D)

   N=size(X,2);
 
   ration1=1;
   Noise1=ration1*randn(D,num)*sd.x+repmat(sd.xmean,1,num);

  
   
   X=[X Noise1];
   
end