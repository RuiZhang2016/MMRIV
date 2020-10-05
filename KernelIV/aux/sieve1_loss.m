function loss = sieve1_loss(df,lambda)
%stage 1 error of KIV

n=length(df.y1);
%m=length(x2);

%stage 1
brac1=make_psd(df.Z1'*df.Z1./n+lambda.*eye(length(df.Z1'*df.Z1)));
beta1=brac1\(df.Z1'*df.X1./n);

%stage 2
X2_hat=df.Z2*beta1;

loss=mse(df.X2,X2_hat);

end

