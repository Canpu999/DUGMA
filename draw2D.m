function P=draw2D(X,Y,R,t)
    
    % draw the scene before registration
    figure
    plot(X(1,:),X(2,:),'.r','MarkerSize', 15)
    hold on
    plot(Y(1,:),Y(2,:),'.b','MarkerSize', 15)    
    title('Before Registration');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');

    % Transform the moving point cloud
    Y=R*Y+t;

    % draw the scene after registration
    figure
    plot(X(1,:),X(2,:),'.r','MarkerSize', 15)
    hold on
    plot(Y(1,:),Y(2,:),'.b','MarkerSize', 15)  
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    title('After Registration');

    P='Finish drawing the images';
end
