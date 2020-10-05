function vx = median_inter(x)
%median interpoint distance

[n,m] = size(x);
%vx = zeros(1,m);
dist = 0;
for i=1:m
    A=repmat(x(:,i),1,n);
    B=A';
    dist= dist+abs(A-B).^2;
end
dist=reshape(dist,[],1);
vx=median(sqrt(dist));
end

