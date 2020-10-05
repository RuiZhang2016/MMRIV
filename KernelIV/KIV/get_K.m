function df=get_K(x,y,z,x_vis)
% precalculate kernel matrices

vx=median_inter(x);
vz=median_inter(z);
[x1, x2, y1, y2, z1, z2]=split(x,y,z,.5);


df.y1=y1;
df.y2=y2;
df.y=y;

df.K_XX=get_K_matrix(x1,x1,vx);
df.K_xx=get_K_matrix(x2,x2,vx);
df.K_xX=get_K_matrix(x2,x1,vx);
df.K_Xtest=get_K_matrix(x1,x_vis,vx);

df.K_ZZ=get_K_matrix(z1,z1,vz);
df.K_Zz=get_K_matrix(z1,z2,vz);

end

