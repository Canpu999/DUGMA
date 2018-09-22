

function  [X, Y, normal] =normalize(x,y)

n=size(x,2);
m=size(y,2);
% calculate the mean
normal.xd=mean(x');
normal.xd=normal.xd';
normal.yd=mean(y');
normal.yd=normal.yd';
%shift 
x=x-repmat(normal.xd,1,n);
y=y-repmat(normal.yd,1,m);
% calculate the scale, a little like the standard deviation
xscale=sqrt(sum(sum(x.^2,2))/n);
yscale=sqrt(sum(sum(y.^2,2))/m);

if xscale>yscale
    normal.scale=xscale;
else
    normal.scale=yscale;
end
X=x/normal.scale;
Y=y/normal.scale;
end



