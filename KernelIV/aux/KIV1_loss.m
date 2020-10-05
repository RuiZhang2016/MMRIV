function loss = KIV1_loss(df,lambda)
%stage 1 error of KIV
%hyp=(lambda,vx,vz)

n=length(df.y1);
m=length(df.y2);

brac=make_psd(df.K_ZZ)+lambda.*eye(n);
gamma=(brac)\df.K_Zz;

loss=trace(df.K_xx-2.*df.K_xX*gamma+gamma'*df.K_XX*gamma)./m;

end

