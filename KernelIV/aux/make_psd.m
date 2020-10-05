function K = make_psd(K)
%for numerical stability, add a small ridge to a symmetric matrix
eps=1e-10;

[N,~]=size(K);
K=(K+K')./2+eps.*eye(N);


end

