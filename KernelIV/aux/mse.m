function err = mse(y,pred)
%GET_PRED_ERROR takes as inputs a vector (y) and its predictions (pred). 
%   it returns a scalar representing the prediction error (err)

N=length(y);
err=norm(y-pred,'fro').^2./N;

end