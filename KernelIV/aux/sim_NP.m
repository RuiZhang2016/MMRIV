function [x,y,z] = sim_NP(f,N)
%SIM_IV generates a data set of size N with according to Newey Powell, with
%structural function f

% simulate (z,x,e)
MU=zeros(3,1);

SIGMA=[1,0.5,0;
       0.5,1,0;
       0,0,1];

r = mvnrnd(MU,SIGMA,N);
u=r(:,1);
t=r(:,2);
w=r(:,3);

x=w+t;
x=normcdf(x./sqrt(2));
z=normcdf(w);
e=u;

% simulate y from f
y=f(x)+e;

end

