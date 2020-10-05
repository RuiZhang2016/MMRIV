function y_pred=KIV_pred(df,hyp,stage)
% returns predictive mean for KIV
% hyp=[lambda,xi]
% stage=(2,3) corresponds to stage 2 and testing

n=length(df.y1);
m=length(df.y2);

lambda=hyp(1);
xi=hyp(2);

brac=make_psd(df.K_ZZ)+lambda.*eye(n);
W=df.K_XX/(brac)*df.K_Zz;
brac2=make_psd(W*W')+m.*xi.*make_psd(df.K_XX);
alpha=brac2\(W*df.y2);

if (stage==2)
    K_Xtest=df.K_XX;
elseif(stage==3)
    K_Xtest=df.K_Xtest;
end

y_pred=(alpha'*K_Xtest)'; %evaluating on xtest

end

