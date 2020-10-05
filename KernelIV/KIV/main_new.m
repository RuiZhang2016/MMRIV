clear;
addpath('../aux'); %test;
addpath('../figures');
addpath('../results'); %test;
warning('off')
diary myDiaryFile
%%
%designs = {'abs' 'linear', 'sin', 'step'};%{'data_8','data_16','data_32'};
designs = cell(4);
sizes = {8,16};
for ii =1:2
    designs{(ii-1)*2+1} = strcat('mendelian_',num2str(sizes{ii}),'_1_0.5');
    designs{(ii-1)*2+2} = strcat('mendelian_',num2str(sizes{ii}),'_0.5_1');
%     designs{(ii-1)*3+3} = strcat('mendelian_',num2str(sizes{ii}),'_2_1');
end
for dataid = 1:4
    design = designs{dataid};
    
    
    %NP, SSG, HLLT
    % [f,sim,x_vis,f_vis]=get_design(design);
    % sample size
    data = load(strcat(design,'.mat'));%load(strcat(design,'_data_2000.mat'));
    x = data.X_train;%cat(1,data.X_train,data.X_dev);
    y = data.Y_train;%cat(1,data.Y_train,data.Y_dev);
    z = data.Z_train;%cat(1,data.Z_train,data.Z_dev);
    z = cast(z,'double');
    x_vis = data.X_test;
    f_vis = data.g_test;
    disp(size(x));disp(size(y));disp(size(z));
    disp(size(x_vis));disp(size(f_vis));

    % simulate data, split into stage 1 and 2 samples
    % f=@(x) abs(x);

    % visualize design
    % plot(data.X_test,data.g_test,'LineWidth',5);
    % figure1 = figure;
    % hold on;
    % scatter(x_vis,f_vis,36,[.3,.3,.3],'filled');
    % scatter(x,y,36,[.5 .5 .5]);
    % xlabel('x','FontSize',20)
    % ylabel('y','FontSize',20)
    % legend({'Structural function','Data'},'Location','southeast','FontSize',20);
    % hold off;
    % saveas(figure1,fullfile('../figures',strcat('design_',design)),'epsc');

    %% KIV - IV, causal validation with lengths simply set
    
    tic
    df=get_K(x,y,z,x_vis);
    toc
    res = zeros(1,10);
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
    
    % visualize estimator
    % figure1 = figure;
    % hold on;
    % scatter(x_vis,f_vis,36,[.3,.3,.3],'filled');
    % scatter(x,y,36,[.5 .5 .5]);
    % scatter(x_vis,y_vis, 48,[0,.6,.6],'filled');
    % xlabel('x','FontSize',20)
    % ylabel('y','FontSize',20)
    % legend({'Structural function','Data','KernelIV'},'Location','southeast','FontSize',20);
    % hold off;
    % saveas(figure1,fullfile('../figures',strcat('KIV_',design)),'epsc');

    % mse
    
    err = mse(y_vis,f_vis);
    res(rep)=err;
    end
    disp(design);
    disp(res);
    disp(mean(res));
    disp(std(res));
end
diary off