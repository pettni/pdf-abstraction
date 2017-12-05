%% \eps,\delta ROBUST CORRECT_BY_DESIGN CONTROL 
% IN THIS WORK, WE GIVE SOME EXPERIMENTAL ROUTINES FOR THE COMPUTATION OF
% \EPS \DELTA ROBUST SPECIFICATIONS.
% Copyleft @ Sofie Haesaert 10/2017
% remark that run_case is an OLD M-file, this is the new one for TACAS.
clear all
close all
clc

%% DEFINE THE SPECIFICATION OF INTEREST
% we currently focus on the following specification 
% F K (first easy spec)
% F^n1 G^n2 K
% K = is a polytope 
n2=3;
DFA.S= 1:n2+1; % set of states
DFA.S0 = [1]; % initial state
DFA.Act = ['k','nk'];
DFA.F = DFA.S(end); % target states
DFA.Trans=[2:DFA.F,DFA.F;ones(1,length(2:DFA.F)),DFA.F]'; % column = control , row = state, value = next state
% https://spot.lrde.epita.fr/trans.html
% 

%% DEFINE THE STOCHASTIC MODEL
m=1;
mw=1;
a=.3;
b=.8;
c=.8;
n=3;
A = [1 -a a
    0 b  0
    0 0   c];
eig(A)
B = [-a*.1;1;0];
Bw=[a*.02;0;.1];
C =[1 0 0];


LTI_concrete = ss(A,[B Bw],C,[],1); % make it a dt state space
K_pol = Polyhedron([-2,2]'); % Ideal following distance
                % (in imaginary metric)
% LTI_abstract.U = Polyhedron([-1,1]);


%% GO FROM 5 DIMENSIONAL TO 1 DIMENSIONAL LTI MODEL
opt = balredOptions('Offset',.001');
[Ml,~, K]=dare(LTI_concrete.A,LTI_concrete.B(:,1),.5*eye(n),0.08)  ;
LTI_concrete_cl = ss(A-B*K,[B Bw],C,[],1); % make it a dt state space

[sysred] = balred(LTI_concrete_cl,1); 

if sysred.c(end)~= 1
    disp(' sysred.c(end)~= 1')
    disp('  sysred.c')
    sysred.c;
    
    T=eye(length(sysred.c));
    T(length(sysred.c),length(sysred.c))= sysred.c(end);
    sysred=ss2ss(sysred,T);
end
    sysred.d=zeros(1,m+mw);
     
rad=.05;
 
LTI_abstract.A= sysred.A;
LTI_abstract.B= sysred.B(:,1);
LTI_abstract.Bw= sysred.B(:,2:end); 
LTI_abstract.C= sysred.C;


%% DETERMINE THE SET TO GRID
LTI_abstract.U = Polyhedron([-.3,.3]);

 [ eps,del,Q,R,P,M ] = epsdel_compute(LTI_concrete,LTI_abstract, Ml,K,rad,.01)
 

LTI_abstract.X = Polyhedron([-10,10]); %[eye(3);-eye(3)],[1,2,3,2,3,4]');



%[QR,P,M,K,eps,deltarange]=App_Bisim(LTI_concrete,sysred, Ml,K);
% FILL IN THE 1 DIMENSIONAL MODEL (PATCH) 
% We pick siams M3 (openloop case)
% A = [0.9951];
% B = [0.1194]; % control input matrix
% Bw = [0.001497 0.01427 0.01467];  % noise disturbance matrix
% C = 1;




 

% TODO: MAKE THIS AUTOMATIC (AS IN SIAM)
%
% PICK DELTA & EPSILON
% + GRIDDING PRECISION


%% Evaluate precision




%%  GO FROM 1 DIMENSIONAL LTI MODEL TO MDP
% BASED ON THE GIVEN GRIDDING PRECISION GRID THE LTI MODEL
% for this we call the gridding function

nu=20;

[MDP,rad] = gridding(LTI_abstract,  rad, nu);
[ eps,del,Q,R,P,M ]= epsdel_compute(LTI_concrete,LTI_abstract, Ml,K,rad,.03)

% Compute Phat
Phat= (P'*M*P)\P'*M;

% with 
% - LTI_abstract = the abstract 1 dimensionsal model
% - X = Polytope of the to be gridded part of the state-space
% - diam = the gridding diameter (euclidean norm for higher dimensions)

%% GO FROM DFA TO NFA, WHOSE TRANSITIONS TAKE AS INPUT THE STATE OF THE MDP
% Now, we need a DFA that represents this. 
% the NFA contains states, initial states, goal states,  transitions
% relations
NFA=  NFA_eps(DFA,eps,MDP,K_pol);




%% FOR THE 1 DIMENSIONAL MODEL COMPUTE THE \DELTA-ROBUST SATISFACTION AND THE POLICY
%  
[p,mu] = del_reach(MDP,NFA,del);

plot(MDP.z_rep,p)





%% Do a simulation with the refined controller

for run=1:10
%1. Initiate
x_2=[2.45;2.5;1.3];
x_1=Phat*x_2;
q=DFA.S0;
N=10;
% start simulation:
for t =1:N
     % compute q based on y_2
     
    q(t+1)=DFA.Trans(q(t),1)*K_pol.contains(C*x_2(:,t)) +DFA.Trans(q(t),1)*(~K_pol.contains(C*x_2(:,t))  );
    [v,maxrep] =min(abs(MDP.z_rep-x_1(:,t)*ones(1,length(MDP.z_rep))));
    u1=mu(q(t+1),maxrep);
    x_1c(:,t) =MDP.z_rep(maxrep);
    u2 = R*u1+Q*x_1c(:,t)-K*(x_2(:,t)-P*x_1(:,t));
    w=randn(1,1);
    x_2(:,t+1)=A*x_2(:,t)+B*u2+Bw*w;
    x_1(:,t+1)=LTI_abstract.A*x_1(:,t)+LTI_abstract.B*u1...
                +LTI_abstract.Bw*w;
    % map x_1 to rep
    [~,maxrep] =min( MDP.z_rep-x_1(:,t+1)*ones(length(MDP.z_rep)));
    
end

% Plot simulation:

plot(1:N,x_1c(1,:),'x')
hold on
plot(1:N+1,x_2(1,:))

end