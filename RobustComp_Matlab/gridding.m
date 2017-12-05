function [MDP,rad] = gridding(LTI,  radiusx, nu)
% GRIDDING computes a MDP based on a LTI system 
%   LTI system with dynamics:
%     LTI.A= A;
%     LTI.B= B;
%     LTI.Bw= Bw;
%     LTI.C= C;
%     LTI.X = polytope of the to be gridded set of states
% diam = diameter between gridding points
%% Unpack LTI 
A = LTI.A;
X = LTI.X;
B = LTI.B;
C = LTI.C;
Bw = LTI.Bw;
U = LTI.U;

%% split gridding diameter into the dimensions
n = size(A,1);
d = radiusx*n^-.5; % nd^2=diam (square grids farthest point from corner to middle is..)
% grid over 2d,


%% Find representative points

% 1. find cube that encompasses X
% check dimension X
if false %X.dim ~= size(A,1) 
    msg = 'X has incompatible.';
    error(msg) 
end


box = X.outerApprox();
boxmax_min= [max(box.V);min(box.V)];

% for each dimension compute grid points
z = cell(n,1);
z_rep = cell(n,1);
for i= 1:n
nz(i)=ceil((boxmax_min(1,i)-boxmax_min(2,i))/(2*d));
z{i} = linspace(boxmax_min(2,i),boxmax_min(1,i),nz(i)+1); % boundaries of the partition sets for first state
z_rep{i} = z{1}(1:end-1) + diff(z{i})/2; % representative points for x1
end
% Add boundaries on the input ???
U_L=min(U.V);
U_H=max(U.V);
u = linspace(U_L,U_H,nu+1); % boundaries of the partition sets for input
u_diam = (U_H-U_L)/nu; % diameter for input
u_rep = u(1:nu) + diff(u)/2; % representative points for input
nu=length(u_rep);



%% 1. Stochastic transitions

if size(A,2) ==1
    
    % simple 1D approach
    P = zeros(nu*nz(1),nz(1)); % empty P matrix
 for k=1:nu
        for i=1:nz(1)
            mean_r = A*[z_rep{1}(i)]+B*u_rep(k);
            t1 = normcdf(z{1},mean_r,(Bw*Bw')^.5);
            P(nz(1)*(k-1)+i,:) = diff(t1);
        end
 end
    

 
else 
    error('not implemented yet')
end
% % transition probability matrices for safety
% P = zeros(nu*nz1*nz2,nz1*nz2);
% for k=1:nu
%     for j=1:nz2
%         for i=1:nz1
%             mean_r = A*[zrep1(i); zrep2(j)]+B_p*urep(k);
%             t1 = normcdf(z1,mean_r(1),1);
%             t2 = normcdf(z2,mean_r(2),1);
%             P(nz1*nz2*(k-1)+nz1*(j-1)+i,:) = reshape((t2(2:nz2+1) - t2(1:nz2))'*(t1(2:nz1+1) - t1(1:nz1)),1,[]);
%         end
%     end
% end



MDP.P=P;
MDP.z_rep=z_rep{1};
MDP.z=z{1};
MDP.y_rep = C*z_rep{1};
MDP.u_rep=u_rep;
MDP.u=u;
rad=d ; % change this for higher dimensions.
end

