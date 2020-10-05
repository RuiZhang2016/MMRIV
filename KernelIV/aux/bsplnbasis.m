%   compute b-spline basis of order r and m interior knots, knots are
%   contained in the vector ktl

function [XX, DX, DDX] = bsplnbasis(x,m,r,kts)

N   = length(x);

%   define the augmented knot set
if nargin==3
    kts = zeros(1,m+2*r-2);
    kts(1:r-1) = zeros(1,r-1);
    kts(m+r:m+2*r-2) = ones(1,r-1);
    kts(r:m+r-1) = 0:1/(m-1):1;
end

%   initialize for recursion
BB  = zeros(N,m+2*r-2,r);
for i = 1:N
    ix = min(find(x(i) >= kts(r:r+m) & x(i) <= kts(r+1:r+m+1)));
    BB(i,ix+r-1,1) = 1;
end

%   recursion
for j = 2:r
    for i = 1:m+2*r-2-j
        if i+j+1 <= m+2*r
            if kts(i+j-1)-kts(i) ~= 0
                a1 = (x-kts(i))/(kts(i+j-1)-kts(i));
            else 
                a1 = zeros(N,1);
            end
            if kts(i+j)-kts(i+1) ~= 0
                a2 = (x-kts(i+1))/(kts(i+j)-kts(i+1));
            else
                a2 = zeros(N,1);
            end
            BB(:,i,j) = a1.*BB(:,i,j-1) + (1-a2).*BB(:,i+1,j-1);
        elseif i+j <= m+2*r
            if kts(i+j)-kts(i) ~= 0
                a1 = (x-kts(i))/(kts(i+j)-kts(i));
            else 
                a1 = zeros(N,1);
            end
            BB(:,i,j) = a1.*BB(:,i,j-1);
        end
    end
end

XX = BB(:,1:m+r-2,r);

if nargout >= 2
    
    DX = zeros(size(XX));
    
    if r > 1
        
        for i = 1:m+r-2
            
            if kts(i+r-1)-kts(i) ~= 0
                a1 = ones(N,1)/(kts(i+r-1)-kts(i));
            else
                a1 = zeros(N,1);
            end
            if kts(i+r)-kts(i+1) ~= 0
                a2 = ones(N,1)/(kts(i+r)-kts(i+1));
            else
                a2 = zeros(N,1);
            end
            
            
            if i < m+r
                DX(:,i) = (r-1)*( a1.*BB(:,i,r-1) - a2.*BB(:,i+1,r-1) );
            else
                DX(:,i) = (r-1)*( a1.*BB(:,i,r-1)  );
            end
        end
        
    end
    
end

if nargout == 3
    
    DDX = zeros(size(XX));
    
    if r > 2
        
        for i = 1:m+r-2
            
            if kts(i+r-1)-kts(i) ~= 0
                a1 = ones(N,1)/(kts(i+r-1)-kts(i));
            else
                a1 = zeros(N,1);
            end
            if kts(i+r)-kts(i+1) ~= 0
                a2 = ones(N,1)/(kts(i+r)-kts(i+1));
            else
                a2 = zeros(N,1);
            end
            if kts(i+r-1)-kts(i+1) ~= 0
                a3 = ones(N,1)/(kts(i+r-1)-kts(i+1));
            else
                a3 = zeros(N,1);
            end
            if kts(i+r-2)-kts(i) ~= 0
                a4 = ones(N,1)/(kts(i+r-2)-kts(i));
            else
                a4 = zeros(N,1);
            end
            if kts(i+r)-kts(i+2) ~= 0
                a5 = ones(N,1)/(kts(i+r)-kts(i+2));
            else
                a5 = zeros(N,1);
            end
            
            DDX(:,i) = (r-1)*(r-2)*( a4.*a1.*BB(:,i,r-2) + a5.*a2.*BB(:,i+2,r-2) - (a1+a2).*a3.*BB(:,i+1,r-2) );
            
        end
        
    end
    
end
    
    
