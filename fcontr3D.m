function [c,ceq]=fcontr3D(X)
   t1=X(1);
   t2=X(2);
   t3=X(3);
   r11= X(4);
   r12= X(5);
   r13=X(6);
   r21=X(7);
   r22=X(8);
   r23=X(9);
   r31=X(10);
   r32=X(11);
   r33=X(12);
   c=[];
   R=[r11 r12 r13; r21 r22 r23; r31 r32 r33];
   ceq(1)=det(R)-1;
   H=R'*R-eye(3);
   ceq(2)=H(1);
      ceq(3)=H(2);
         ceq(4)=H(3);
            ceq(5)=H(4);
               ceq(6)=H(5);
                  ceq(7)=H(6);
                     ceq(8)=H(7);
                        ceq(9)=H(8);
                           ceq(10)=H(9);

end
