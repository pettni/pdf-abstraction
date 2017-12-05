% Decouple Noise

% standard deviation of the process noise
Zigma = Bw*Bw';
det(Zigma)
L = sqrtm(inv(Zigma));
Linv = inv(L);

A_p  = L*A/L;
B_p = L*B;
%Ql=Q/L;
%===================================================================


% state space transformation
% obtaining the boundaries
Safe_L= L*Safe2D;
coef1=1;coef2=1;
A_L=min(Safe_err.V);
A_H=min(Safe_err.V);
% gridding bounds
Bound_L =...% max([min(Safe_L.V)';[-3,-3]],[],2);
-abs([coef1*A_L/Linv(2,1); coef2*A_L/Linv(2,2)])
Bound_H =... min([max(Safe_L.V)';[3,3]],[],2); 
abs([coef1*A_H/Linv(2,1); coef2*A_H/Linv(2,2)])


disp('Lipschitz constants')
Hz1 = (abs(A_p(1,1))+abs(A_p(2,1)))*2/sqrt(2*pi)
Hz2 = (abs(A_p(1,2))+abs(A_p(2,2)))*2/sqrt(2*pi)
Hup = (abs(B_p(1,1))+abs(B_p(2,1)))*2/sqrt(2*pi)
disp('Optimal gridding size')
Error=.1;
delta_z1 = Error/(2*N*Hz1)
delta_z2 = Error/(2*N*Hz2)
delta_up = Error/(2*N*Hup)

%% tune these partition diameters

% start from max number of finite state
 
%delta_z1 = 0.001;
%delta_z2 = 0.005;
delta_z1 = 0.0011;
delta_z2 = 0.0015;
%delta_up = 0.1;
nu=50;


Error = N*(Hz1*delta_z1 + Hz2*delta_z2)% (simulation error, not bisimulatio)

% Gridding dimension 1
nz1 = ceil((Bound_H(1)-Bound_L(1))/delta_z1); % number of partition sets for x1
z1 = linspace(Bound_L(1),Bound_H(1),nz1+1); % boundaries of the partition sets for first state
delta_z1 = (Bound_H(1)-Bound_L(1))/nz1 % diameter for x1
zrep1 = z1(1:nz1) + delta_z1/2; % representative points for x1

% Gridding dimension 2
nz2 = ceil((Bound_H(2)-Bound_L(2))/delta_z2); % number of partition sets for x2
z2 = linspace(Bound_L(2),Bound_H(2),nz2+1); % boundaries of the partition sets for state
delta_z2 = (Bound_H(2)-Bound_L(2))/nz2 % diameter for x2
zrep2 = z2(1:nz2) + delta_z2/2; % representative points for x2

if nz2*nz1>10^7
display('nz2*nz1>10^7')
nz2*nz1
pause
end

nz2*nz1
pause
 U_L=min(U_Lt_cor);
U_H=max(U_Ht_cor);
u = linspace(U_L,U_H,nu+1); % boundaries of the partition sets for input
delta_up = (U_H-U_L)/nu % diameter for input

if delta_up>.2
   
    display('delta_up>.2')
 
    pause
end
urep = u(1:nu) + delta_up/2; % representative points for input

%% 1. Stochastic transitions


% % 1. create P_1=(Pz2 \otime ones(?))*(ones(?) \otime Pz1)

     Pz1=zeros(nz1,nz1);
     for i=1:nz1 
            t1 = normpdf(zrep1,zrep1(i),1)*delta_z1; 
            Pz1(i,:) =  t1 ;
     end
        Pz2=zeros(nz2,nz2);
    for j=1:nz2 
            t2 = normpdf(zrep2,zrep2(j),1)*delta_z2;
            Pz2(j,:) =  t2 ;
    end
 
%Ptry=kron(Pz1,Pz2);
%Ptry2=kron(Pz1,eye(nz2))*kron(eye(nz1),Pz2); % (Best memory wise ?)

%% 2. Determinsitic transition

%% create sparse matrix 
[Pdet,zrepind]=Det_trans(A_p,B_p,zrep1,zrep2,urep,L,Safe2D);
nstates=length(zrepind);
display('sparse ready')
% (prune sparse matrix?)

%% Create indicator function
 
Vind=double(Safe2D.contains(L\[kron(zrep1,ones(1,nz2));kron(ones(1,nz1),zrep2)])'); 
disp('composed indicator function, ready for iterations')


Vsq=zeros(nz2,nz1);
Vsq(:)=Vind;
 V_aux=zeros(nz1*nz2,nu);
  figure,
    Policy=zeros(nz2,nz1);figure ; hold on;
for k=N:-1:1
  
    
     Vsq(:)=Pz2*Vsq*Pz1'; 
    %V=Pdet*Vsq(:);
    V_aux(:) =  Pdet*Vsq(:);
    
        % find allowed actions
  % zrepind  safe states only
  % 
  % [nz2,1]*zrepind-nz2
 
    %1. create safe set indices
%        Allact=sparse( kron(([nz2,1]*zrepind-nz2)',ones(nu,1)),kron( ones(nstates,1),(1:nu)'),...
%           ((kron(ones(nstates,1),urep')- kron((Q*zrepind)',ones(nu,1)))>=U_Lt_cor(k)) & ( (kron(ones(nstates,1),urep')- kron((Q*zrepind)',ones(nu,1)))<=U_Ht_cor(k)),... 
 
disp(k)

  %  Vsq(:) = Vind.*max(V_aux(:,urep>=U_Lt_cor(k) & urep<=U_Ht_cor(k)),[],2);
    [Vsq(:),Policy(:)] =  max(V_aux,[],2);
  % mesh(zrep1,zrep2,Vsq); hold on
  %  Vsq(:) =  max(Allact.*V_aux,[],2);
 savefile = ['Policy',num2str(k),'.mat']; 

  var_save=['''Policy''',',','''Vsq'''];
   eval([' save( savefile ', ',',var_save,')'])
%mesh(zrep1,zrep2,Vsq);drawnow;
end
disp('done with iterations, Got a value function')
figure, mesh(zrep1,zrep2,Vsq),


