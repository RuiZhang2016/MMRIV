function [x,y,z] = sim_HLLT(f,N,rho)
%SIM_IV generates a data set of size N with according to Hartford, Lewis,
%Leyton-Brown, Taddy with structural function f and confoundedness rho

% simulate (z,t,s,v)
z=normrnd(0,1,N,1);
v=normrnd(0,1,N,1);
s=unidrnd(7,N,1);
t=rand(N,1).*10;

% simulate e from v
e=normrnd(rho.*v,1-rho.^2,N,1);

% simulate p from z,t,v
p=25+(z+3).*get_psi(t)+v;

% simulate y from f
y=f(p,t,s)+e;
x=[p t s];

end

