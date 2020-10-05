function y_vis = sim_pred(design,N)
%simulate Newey Powell problem, return y_vis for KIV

% simulate data for this design
[f,sim,x_vis,~]=get_design(design);
[x,y,z]=sim(f,N);
df=get_K(x,y,z,x_vis);

% initialize hyperparameters for tuning
lambda_0=log(0.05); %hyp1=lambda. log(0.05) for NP
xi_0=log(0.05); %hyp2=xi. log(0.05) for NP

% stage 1 tuning
KIV1_obj=@(lambda) KIV1_loss(df,exp(lambda)); %exp to ensure pos; fixed vx,vz
lambda_star=fminunc(KIV1_obj,lambda_0);

% stage 2 tuning
KIV2_obj=@(xi) KIV2_loss(df,[exp(lambda_star),exp(xi)]); %exp to ensure pos
xi_star=fminunc(KIV2_obj,xi_0);

% evaluate on full sample using tuned hyperparameters
y_vis=KIV_pred(df,[exp(lambda_star),exp(xi_star)],3); %exp to ensure pos

end

