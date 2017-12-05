function [Pdet,zrepind]=Det_trans(A_p,B_p,zrep1,zrep2,urep,L,Safe2D)

delta_z1=zrep1(2)-zrep1(1);
delta_z2=zrep2(2)-zrep2(1);

nz1=length(zrep1);
nz2=length(zrep2);
nu=length(urep);

zrepindt=[kron(floor(zrep1/delta_z1+delta_z1/2),ones(1,nz2));kron(ones(1,nz1),floor(zrep2/delta_z2+delta_z2/2))];
zrepind=zrepindt(:,Safe2D.contains(L\[kron(zrep1,ones(1,nz2));kron(ones(1,nz1),zrep2)]));
nstates=length(zrepind);
zrepind=zrepind-(min(zrepindt,[],2))*ones(1,nstates)+ones(2,nstates);
disp('collected all safe states')
%Zrep=[kron(zrep1,ones(1,nz2));kron(ones(1,nz1),zrep2)];
Pdet=sparse(nz1*nz2*nu,nz1*nz2);
for k=1:nu;

Zrepnext= A_p* [zrep1(zrepind(1,:));zrep2(zrepind(2,:))]+B_p*urep(k)*ones(1,nstates); 
Zrepn_index=floor([delta_z1^-1,0;0,delta_z2^-1]*Zrepnext+[delta_z1/2;delta_z2/2]*ones(1,nstates))...
                -(min(zrepindt,[],2))*ones(1,nstates)+ones(2,nstates); 
            
Zrepred=[nz2,1]*zrepind(:,Zrepn_index(1,:)<=nz1&Zrepn_index(2,:)<=nz2&Zrepn_index(1,:)>0&Zrepn_index(2,:)>0)-nz2;
Znextred=[nz2,1]*Zrepn_index(:,Zrepn_index(1,:)<=nz1&Zrepn_index(2,:)<=nz2&Zrepn_index(1,:)>0&Zrepn_index(2,:)>0)-nz2;

Pdet=Pdet+sparse(nz1*nz2*(k-1)+Zrepred,Znextred,ones(size(Zrepred)),nz1*nz2*nu,nz1*nz2);

end


end