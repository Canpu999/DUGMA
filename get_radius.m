%  get the standard deviation for the x,y

function  [sd] =get_radius(x)


n=size(x,2);

% calculate the mean
normal.xd=mean(x');
normal.xd=normal.xd';

%shift 
x=x-repmat(normal.xd,1,n);

% calculate the scale, a little like the standard deviation
xscale=sqrt(sum(sum(x.^2,2))/n);

sd.x=xscale;

sd.xmean=normal.xd;


end



