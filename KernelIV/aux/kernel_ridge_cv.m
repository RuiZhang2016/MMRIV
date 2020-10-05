function mse_total = kernel_ridge_cv(df,lambda)
%cross validation error of kernel ridge regression with 2 folds
%hyp=(lambda)

y2_pred=kernel_ridge_pred(df,lambda,1);
mse2=mse(df.y2,y2_pred);
y1_pred=kernel_ridge_pred(df,lambda,2);
mse1=mse(df.y1,y1_pred);
mse_total=mse1+mse2;

end

