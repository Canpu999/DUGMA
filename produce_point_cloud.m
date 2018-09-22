function [X,X_C, Y, Y_C, R_G, t_G]=produce_point_cloud(rand_degree, rand_noise, rand_outliers, rand_miss_rate, shape, St_data, D)
   
                                    if D==2
                                         degreeX_temp=rand_degree*(2*rand-1);
                                    else
                                         degreeX_temp=rand_degree*(2*rand-1);
                                         degreeY_temp=rand_degree*(2*rand-1);
                                         degreeZ_temp=rand_degree*(2*rand-1);                            
                                    end
                                    noise_temp=rand_noise*rand;
                                    outliers_temp=round(rand_outliers*rand);  
                                    miss_rate_temp=rand_miss_rate*rand;
                                    
                                    % produce the ground truth 
                                    if D==2
                                        R_G=[cosd(degreeX_temp) -sind(degreeX_temp);sind(degreeX_temp) cosd(degreeX_temp)];
                                        t_G=(1-2*rand(2,1))*0.5;
                                    else
                                        R_G=eul2rotm([degreeZ_temp,degreeY_temp,degreeX_temp]/180*pi);
                                        t_G=(1-2*rand(3,1))*0.5;
                                    end

                                    % load the model
                                    if D==2
                                         load([St_data,num2str(shape),'.mat']);   
                                    else
                                         load([St_data,num2str(shape),'.mat']);     
                                    end
                                    
                                    
                                    Model=X;
                                    Model=double(X);
                                    
                                    % add outliers to model
                                    sd=get_radius(Model); 
                                    Model_X=add_outliers(Model,sd,outliers_temp,D);  % model for point cloud X
                                    Model_Y=add_outliers(Model,sd,outliers_temp,D);  % model for point cloud Y
                                        
                                    
                                    % add noise to the model and take samples from the model for X,Y 
                                    if D==2
                                        [X,Y_fake,X_C,Y_C_fake]=noise_and_sampling2D(Model_X,R_G,t_G,noise_temp,miss_rate_temp);
                                        [X_fake,Y,X_C_fake,Y_C]=noise_and_sampling2D(Model_Y,R_G,t_G,noise_temp,miss_rate_temp);
                                    else
                                        [X,Y_fake,X_C,Y_C_fake]=noise_and_sampling3D(Model_X,R_G,t_G,noise_temp,miss_rate_temp);
                                        [X_fake,Y,X_C_fake,Y_C]=noise_and_sampling3D(Model_Y,R_G,t_G,noise_temp,miss_rate_temp);
                                    end
                                    % Ground truth
                                    t_G=-pinv(R_G)*t_G;
                                    R_G=pinv(R_G);

end