%%%%%%%%%%%%%%%%%%%%%%%%%% Interface function of DUGMA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%input parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% X is the fixed point cloud
% X_C is the covariance of X
% Y is the moving point cloud
% Y_C is the covariance of Y
% R0 is the intial rotation matrix
% t0 is the initial translation matrix
% Max_Iteration is the maximum iteration
% accuracy_rotation is the accuracy threshold for the rotation matrix
% accuracy_translation is the accuracy threshold for the translation matrix
% accuracy_sig is the accuracy threshold for the minimum distance

%%%%%%%%%%%%output parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# R is the estimated rotation matrix by DUGMA
# t is the estimated translation matrix by DUGMA
 
function [R,t]=DUGMA(X,Y,X_C,Y_C,R0,t0,Max_Iteration, accuracy_rotation, accuracy_translation,accuracy_sig)   
      % A is the coefficient for the energy function
      global A
      warning('off','all')
      % Convert the data type
      X=double(X);
      X_C=double(X_C);
      Y=double(Y);
      Y_C=double(Y_C);
      X0=X;
      Y0=Y;

      % normalize
      [X, Y, normal] =normalize(X,Y);

      % get the number and dimension
      N=size(X,2);
      M=size(Y,2);
      D=size(X,1);

      % initial rotation and translation and sigma
      R=R0;
      t=t0;

      % initialize the dataset on the gpu
      [Success_Initialize,ad0,ad1,ad2,ad3,ad4,ad5]=init(X,X_C,Y,Y_C);

      %update_sig_old(X,R*Y+t) on the gpu
      [sig,ad0,ad1,ad2,ad3,ad4,ad5]=update_sig(ad0,ad1,ad2,ad3,ad4,ad5,R,t,N,M,D);


      % the angle hasn't converge and we need iterate again
      con_angle=true;
 
      % Initialization
      iteration=1;
      angle=zeros(2*D-3,1)+10000;
      options = optimset('Display', 'off') ;
      while ((iteration<=Max_Iteration) && (con_angle || (abs((t0-t)'*(t0-t))>accuracy_translation)) && (sig>accuracy_sig))
              time_start=tic;
              % calculate the coefficient
              [Success_Coefficient,ad0,ad1,ad2,ad3,ad4,ad5]=compi(ad0,ad1,ad2,ad3,ad4,ad5,R,t,N,M,D,sig);
              A=Success_Coefficient';

              % solve the R,t
                  if D==2
                     [dT fval exitflag]=fmincon('energy2D',[0 0 1 0],[],[],[],[],[],[],'fcontr2D',options);
                     % Update the rotation and translation
                     R=[dT(3) -dT(4);dT(4) dT(3)]*R;
                     t0=t;
                     t=[dT(1);dT(2)]+t;
                     angle0=angle;
                     angle=asind(R(2));                                
                  else
                     [dT,fval,exitflag]=fmincon('energy3D',[0 0 0 1 0 0 0 1 0 0 0 1],[],[],[],[],[],[],'fcontr3D',options);
                     % Update the rotation and translation
                     dt=dT(1:D);
                     t0=t;
                     t=dt'+t;
                     R=[dT(4),dT(5),dT(6);dT(7),dT(8),dT(9);dT(10),dT(11),dT(12)]*R;
                     angle0=angle;
                     angle=rotm2eul(R);                   
                  end
                  
                  % check whether it can opimized 
                  if exitflag<0
                     Error='The Energy Function Can Not Be Converged!'
                     break;
                  end

                 % check whether the angle has converged
                 if (D==2)                        
                     if (abs(angle0-angle)>accuracy_rotation)
                         con_angle=true;
                     else
                         con_angle=false; 
                     end
                 else
                     if (abs(angle0(1)-angle(1))>accuracy_rotation) ||  (abs(angle0(2)-angle(2))>accuracy_rotation) || (abs(angle0(3)-angle(3))>accuracy_rotation)
                         con_angle=true;
                     else
                         con_angle=false;
                     end
                 end

                 % Update the sig
                 [sig,ad0,ad1,ad2,ad3,ad4,ad5]=update_sig(ad0,ad1,ad2,ad3,ad4,ad5,R,t,N,M,D);

                 % display the process information
                 time_end=toc(time_start);
                 disp(['Iteration=',num2str(iteration),' Sigma=',num2str(sig),' Time=',num2str(time_end),'s']);
                 
                 % update the iteration
                 iteration=iteration+1;
      end

      % check whether the solution is reliable 
      if exitflag<0
          R=eye(D);
          t=zeros(D,1);
      else                        
          % denormalize
          R=R;
          t=normal.xd+t*normal.scale-R*normal.yd;
      end

      % Free the memory
      Success_Free_Memory=fr_m(ad0,ad1,ad2,ad3,ad4,ad5); 
      clear ad0 ad1 ad2 ad3 ad4 ad5; 
      
      % Draw the image before and after registration
      if  D==2
          draw2D(X0,Y0,R,t);
      else
          draw3D(X0,Y0,R,t)
      end
      
      
end


