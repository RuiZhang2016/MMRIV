clear;
addpath('../aux'); %test;
addpath('../figures');
addpath('../results'); %test;
warning('off')
diary myDiaryFile
%%
designs = {'mendelian_8_1_1', 'mendelian_16_1_1', 'mendelian_32_1_1',...
    'mendelian_16_1_0.5', 'mendelian_16_1_2',...
    'mendelian_16_0.5_1', 'mendelian_16_2_1'}; % mendelian_{#iv}_{c2}_{c1}
for dataid = 1:7
    design = designs{dataid};
    data = load(strcat(design,'.mat'));
    x = data.X_train;
    y = data.Y_train;
    z = data.Z_train;
    z = cast(z,'double');
    x_vis = data.X_test;
    f_vis = data.g_test;
    disp(size(x));disp(size(y));disp(size(z));
    disp(size(x_vis));disp(size(f_vis));

    %% KIV - IV, causal validation with lengths simply set
    
    tic
    df=get_K(x,y,z,x_vis);
    toc
    res = zeros(1,1);
    for rep = 1:1
        % initialize hyperparameters for tuning
        lambda_0=log(0.05); %hyp1=lambda. log(0.05) for NP
        xi_0=log(0.05); %hyp2=xi. log(0.05) for NP

        % stage 1 tuning
        KIV1_obj=@(lambda) KIV1_loss(df,exp(lambda)); %exp to ensure pos; fixed vx,vz
        lambda_star=fminunc(KIV1_obj,lambda_0);

        % stage 2 tuning
        KIV2_obj=@(xi) KIV2_loss(df,[exp(lambda_star),exp(xi)]); %exp to ensure pos
        xi_star=fminunc(KIV2_obj,xi_0);
        tic
        % evaluate on full sample using tuned hyperparameters
        y_vis=KIV_pred(df,[exp(lambda_star),exp(xi_star)],3); %exp to ensure pos
        toc

        err = mse(y_vis,f_vis);
        res(rep)=err;
    end
    disp(design);
    disp(res);
    disp(mean(res));
    disp(std(res));
end
diary off