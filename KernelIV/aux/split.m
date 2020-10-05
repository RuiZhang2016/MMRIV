function [x1, x2, y1, y2, z1, z2]=split(x,y,z,frac)
%sample splitting

N=length(y);
n=round(frac.*N);

x1=x(1:n,:);
x2=x(n+1:N,:);
y1=y(1:n);
y2=y(n+1:N);
z1=z(1:n,:);
z2=z(n+1:N,:);

end

