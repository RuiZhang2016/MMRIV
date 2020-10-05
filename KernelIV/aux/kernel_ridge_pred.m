function y_pred =kernel_ridge_pred(df,lambda,stage)
% returns predictive mean for GP regression
% stage (1,2,3) corresponds to (cv1,cv2,test)

if (stage==1)
    K_xX=(df.K12)';
    K_XX=df.K11;
    y=df.y1;
    n=length(y);
elseif (stage==2)
    K_xX=df.K12;
    K_XX=df.K22;
    y=df.y2;
    n=length(y);
elseif (stage==3)
    K_xX=df.Kta;
    K_XX=df.Kaa;
    y=df.y;
    n=length(y);
end

brac=make_psd(K_XX)+lambda.*eye(n);
y_pred=K_xX/(brac)*y;

end