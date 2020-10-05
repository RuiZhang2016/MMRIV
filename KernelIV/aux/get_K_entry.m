function K_entry = get_K_entry(x,z,v)
%GET_K_ENTRY calculates an entry of the K matrix for inputs x and z. It
%   uses a radial basis function kernel with width v

K_entry=exp((norm((x-z)./v).^2)./(-2));

end

