function y_pred =sieveIV_pred(df,hyp,stage)
% returns sieve IV pred

n=length(df.y1);
m=length(df.y2);

lambda=hyp(1);
xi=hyp(2);

%stage 1
brac1=make_psd(df.Z1'*df.Z1./n+lambda.*eye(length(df.Z1'*df.Z1)));
beta1=brac1\(df.Z1'*df.X1./n);

%stage 2
X2_hat=df.Z2*beta1;
brac2=make_psd(X2_hat'*X2_hat./m+xi.*eye(length(X2_hat'*X2_hat)));
beta2=brac2\(X2_hat'*df.y2./m);

if (stage==2)
    Xtest=df.X1;
elseif(stage==3)
    Xtest=df.X_vis;
end

y_pred=Xtest*beta2;

end