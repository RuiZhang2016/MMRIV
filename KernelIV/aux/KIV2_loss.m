function loss=KIV2_loss(df,hyp)
%stage 2 error of KIV

y1_pred=KIV_pred(df,hyp,2); %third argument: xtest=x1 in causal validation
loss=mse(df.y1,y1_pred);

end

