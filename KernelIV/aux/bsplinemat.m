function [XX, DX, DDX] = bsplinemat(x,m,r,kts)

%   r-order B spline basis with m interior knots (total dimension
%   is J=m+r), with knots evenly-spaced between lo and hi

m  = m + 2;

if nargin == 4

    %   compute basis
    if nargout == 1
        XX= bsplnbasis(x,m,r,kts);
    elseif nargout == 2
        [XX, DX] = bsplnbasis(x,m,r,kts);
    elseif nargout == 3
        [XX, DX, DDX] = bsplnbasis(x,m,r,kts);
    end

else
    
    %   compute basis
    if nargout == 1
        XX= bsplnbasis(x,m,r);
    elseif nargout == 2
        [XX, DX] = bsplnbasis(x,m,r);
    elseif nargout == 3
        [XX, DX, DDX] = bsplnbasis(x,m,r);
    end

end