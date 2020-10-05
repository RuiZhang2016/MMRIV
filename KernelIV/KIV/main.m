clear;
addpath('../aux'); %test;
addpath('../figures');
addpath('../results'); %test;

%%

design='NP'; %NP, SSG, HLLT
[f,sim,x_vis,f_vis]=get_design(design);
% sample size
N=1000;

% simulate data, split into stage 1 and 2 samples
rng('default');
[x,y,z]=sim(f,N);
disp(size(x));
disp(size(y));
disp(size(z));

% visualize design
plot(x_vis,f_vis,'LineWidth',5);
hold on;
scatter(x,y,36,[.5 .5 .5]);
xlabel('x','FontSize',20)
ylabel('y','FontSize',20)
legend({'Structural function','Data'},'Location','southeast','FontSize',20);
hold off;
saveas(gcf,fullfile('../figures',strcat('design_',design)),'epsc');

%% KIV - IV, causal validation with lengths simply set

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

% visualize estimator
plot(x_vis,f_vis,'LineWidth',5);
hold on;
scatter(x,y,36,[.5 .5 .5]);
plot(x_vis,y_vis,'--r','LineWidth',5);
xlabel('x','FontSize',20)
ylabel('y','FontSize',20)
legend({'Structural function','Data','KernelIV'},'Location','southeast','FontSize',20);
hold off;
saveas(gcf,fullfile('../figures',strcat('KIV_',design)),'epsc');

% mse
disp('mse:');
disp(mse(y_vis,f_vis));

%% KIV - 100 simulations

clear;
rng('default')
design='NP'; %NP, SSG, HLLT
[f,sim,x_vis,f_vis]=get_design(design);

% sample size
N=1000;
n_trials=100;
results=zeros(n_trials,N);
results_mse=zeros(n_trials,1);

for i=1:n_trials
    if mod(i,10)==0
        disp(num2str(i));
    end
    results(i,:)= sim_pred(design,N)';
    results_mse(i)=mse(results(i,:)',f_vis);
end

% visualize 100 samples
means=mean(results);
sorted=sort(results);
fifth=sorted(5,:);
nintetyfifth=sorted(95,:);

plot(x_vis,f_vis,'LineWidth',2);
hold on;
plot(x_vis,means,'--r','LineWidth',2);
plot(x_vis,fifth,':r','LineWidth',2);
plot(x_vis,nintetyfifth,':r','LineWidth',2);
xlabel('x');
ylabel('y');
hold off;
saveas(gcf,fullfile('../figures',strcat('KIV_',design,'_',num2str(n_trials))),'epsc');