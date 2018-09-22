function P=draw3D(X,Y,R,t)

    % draw the scene before registration
    figure
    plot3(X(1,:),X(2,:),X(3,:),'.r','MarkerSize', 15)
    hold on
    plot3(Y(1,:),Y(2,:),Y(3,:),'.b','MarkerSize', 15)
    title('Before Registration');
    xlabel('X Coordinate');
    ylabel('Y Coordinate');

    % transform the moving point cloud
    Y=R*Y+t;

    % draw the scene after registration
    figure
    plot3(X(1,:),X(2,:),X(3,:),'.r','MarkerSize', 15)
    hold on
    plot3(Y(1,:),Y(2,:),Y(3,:),'.b','MarkerSize', 15)   
    xlabel('X Coordinate');
    ylabel('Y Coordinate');
    title('After Registration');

    P='Finish drawing the images';
end