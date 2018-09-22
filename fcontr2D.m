function [c d]=fcontr2D(X)
  t1=X(1);
  t2=X(2);
  r11=X(3);
  r21=X(4);
  c=[];
  d=r11^2+r21^2-1;

end
