function K_true=get_K_matrix(X1,X2,v)
%GET_K_MATRIX returns the covariance matrix for the noiseless GP with
%   radial basis function kernel at inputs X1 and X2

M=length(X1);
N=length(X2);
K_true=zeros(M,N);

for i=1:M
    for j=1:N
        K_true(i,j)=get_K_entry(X1(i,:)',X2(j,:)',v);
    end
end
        
end

