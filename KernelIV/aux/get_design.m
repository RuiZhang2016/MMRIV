function [f,sim,x_vis,f_vis]=get_design(design)
% gives simulation objects for the chosen design
% f - structural function
% sim - simulation generator
% x_vis - x values for visualizing the estimated function
% f_vis - f values for visualizing the structural function

if strcmp(design,'NP')
    f=@(x) log(abs(16.*x-8)+1).*sign(x-0.5); % newey powell
    sim=@(f,N) sim_NP(f,N);
    x_vis=linspace(-0,1,1000)'; %1000 test points
    f_vis=f(x_vis);
elseif strcmp(design,'HLLT')
    f=@(p,t,s) 100+(10+p).*s.*get_psi(t)-2.*p; % Hartford Lewis Leyton-Brown Taddy
    sim=@(f,N,rho) sim_HLLT(f,N,rho);
    p_vis=linspace(2.5,14.5,20)'; 
    t_vis=linspace(-0,10,20)';
    s_vis=[1 2 3 4 5 6 7]';
    [P,T,S]=meshgrid(p_vis,t_vis,s_vis);
    x_vis=[reshape(P,[],1) reshape(T,[],1) reshape(S,[],1)]; %20*20*7 test points
    f_vis=f(x_vis(:,1),x_vis(:,2),x_vis(:,3));
elseif strcmp(design,'CC')
    f=@(x) 4.*x-2; % chen christensen
    sim=@(f,N) sim_NP(f,N);
    x_vis=linspace(-0,1,1000)'; %1000 test points
    f_vis=f(x_vis);
else 
    disp('choose valid design')
end
%disp('chosen design:')
%disp(design)

end

