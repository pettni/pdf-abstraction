%%%The final Draft of the case study
clear all

%% Central intialise
% amount of output to generate
allplots=0;
printmat=@(x) fprintf([repmat('%8.4g &',1, size(x,2)-1),'%8.4g \\\\ \n'],x' );

%% Generate model
echo off

CreateModel;echo off
% Output = sysdfull
display(' Original model' )
fprintf('A= ');printmat(sysdfull.a);fprintf('\n')
fprintf('B= ');printmat(sysdfull.b(:,1));fprintf('\n')
fprintf('Bw= ');printmat(sysdfull.b(:,2:end));fprintf('\n')
fprintf('C= ');printmat(sysdfull.c);fprintf('\n')

echo on
%% Reduce model 
echo off
[Ml,~, F]=dare(sysdfull.A,sysdfull.B(:,1), sysdfull.C'*sysdfull.C,0.02)  ;
[Ml,~, K]=dare(sysdfull.A,sysdfull.B(:,1),sysdfull.C'*sysdfull.C+ .5*eye(5),0.008)  ;

sysdclosed=ss(sysdfull.a-sysdfull.b(:,1)*F,sysdfull.b(:,1:end),sysdfull.c,sysdfull.d(:,1:end),-1); %(ignore disturbance)
sysred=balred(sysdclosed,2);
%sysred=balred(sysdfull,2);
 sysred=ss(tf(sysred));
 
 if sysred.c(end)~= 1
  
     disp(' sysred.c(end)~= 1')
    disp('  sysred.c')
    sysred.c;
    
   T=eye(length(sysred.c));
   T(length(sysred.c),length(sysred.c))= sysred.c(end);
   sysred=ss2ss(sysred,T);
    sysred.d=zeros(1,4);
       pause
 end
     sysred.d=zeros(1,4);
display(' Reduced model' )
% print 
fprintf('A2= ');printmat(sysred.a);fprintf('\n')
fprintf('B2= ');printmat(sysred.b(:,1));fprintf('\n')
fprintf('Bw2= ');printmat(sysred.b(:,2:end));fprintf('\n')
fprintf('C2= ');printmat(sysred.c);fprintf('\n')

% Compute Approximate simulation relation
[Q,R,P,M,K,eps,deltarange]=App_Bisim(sysdfull,sysred, Ml,K,60);
Epsilon=eps;%(end-5);
del=deltarange;%(end-5);

%% Abstract reduced model to finite state model 
% interface is 
% u=R*us+Q*xs+ K(x-Pxs)

% Find abstraction of closed loop system 
%  x=(As-BQ)x+But
% with us=-Q x+ ut
% new interface
% u=Rut+K(...)

B = sysred.B(:,1)*R^-1;
A = sysred.A;
Bw =sysred.B(:,2:end);
C =sysred.C;


% Safe set 2D 

ubound=Polyhedron([-100,100]');% input bound
% correct
Safe=Polyhedron([-.5,.5]'); % temperature range
Safe_err=Safe-Polyhedron([-Epsilon,Epsilon]');
% 2D safe set
Safe2D=ubound*Safe_err;


% Time dependend bound on input 
%(us\in[min_us max_us])
u_com=.2;
umax=1;
umin=0;
predicted_heating= [.5,.5,.5,.5,.5,.5];
U_Lt= -predicted_heating;
U_Ht=1-predicted_heating;
u_d= sqrt(K/M*K')*Epsilon; % .63 % (comes from K(x-x), sqrt(K/MK')*eps)
u_d=0;
if u_d>.5
  display(' u_d>.5')
  pause 
end


% correct with R; and with u_d 
U_Lt_cor=max([R*( U_Lt+u_d);-u_com*ones(size(U_Ht))]);
U_Ht_cor=min([R*( U_Ht-u_d);u_com*ones(size(U_Ht))]);

% horizon of the spec
N = 6;
Faust_mini
%figure, mesh(zrep1,zrep2,Vsq),
%  
% F_z=scatteredInterpolant([kron(zrep1,ones(1,nz2));kron(ones(1,nz1),zrep2)]',Vsq(:),'nearest');
% x1=(Bound_L(1):.05:Bound_H(1));
% x2=(Bound_L(2):.05:Bound_H(2));
% [X1,X2]=meshgrid(x1,x2);
% figure, mesh(X1,X2,F_z(X1,X2))
% title('safety probabilities')
% 
% 
% X=L\[kron(zrep1,ones(1,nz2));kron(ones(1,nz1),zrep2)];
% 
% F_q=scatteredInterpolant(X',Vsq(:),'nearest');
% x1=(-1:.1:1);
% x2=(-.6:0.1:.6);
% [X1,X2]=meshgrid(x1,x2);
% figure, mesh(X1,X2,F_q(X1,X2))
% note that this is still a transformed set,

% % bias of input due to current state us=u+B*Q
% us=u+-Qxs
% u\in [U_Lt_cor,U_Ht_cor]+ Qxs
%
 
%% 
% Also construct policies
% Policy{1,2,3,4,...}



%% Then do a simulation 

Monte=100000; 

A=sysdfull.a;B=sysdfull.b(:,1);Bw=sysdfull.b(:,2:end);C=sysdfull.c; 

Aa=sysred.a;Ba=sysred.b(:,1);Bwa=sysred.b(:,2:end); Ca=sysred.c;


%  make a policy function via scattered  $&$ nearest
X=L\[kron(zrep1,ones(1,nz2));kron(ones(1,nz1),zrep2)];



%%% CASE 1 in simulation example paper
% x= zeros(5,Monte);
% xa= zeros(2,Monte);
% xun= zeros(5,Monte);
 
%%
x=P*(L\[z1(1);z2(nz2/2)])*ones(1,Monte);%P*(L\[z1(1);z2(1)])*ones(1,Monte);
xa=P\x;%zeros(2,Monte);
xun=x;%zeros(5,Monte);
x=x-[0;0.201;0;0;0]*ones(1,Monte);
%x=(P*(L\[z1(1);z2(nz2/2)])+V(2,:)'*.95)*ones(1,Monte);%P*(L\[z1(1);z2(1)])*ones(1,Monte);

% %    
% %  

   
Y=zeros(7,Monte);

Yun=zeros(7,Monte);

savefile = ['Policy',num2str(6),'.mat'];  
load(savefile,'Policy')
 rng('default')
Y(1,:)=C*x; 
Yun(t,:)=Ca*xa; 
%Yun(1,:)=C*xun;
for t=1:6
   
%    Pol=scatteredInterpolant(X',urep(Policy(:))','nearest');
    w=randn(3,Monte);
    %% Find control
    % from x_a go to abstract model
indices= [nz2,1]*max(min((floor([delta_z1^-1,0;0,delta_z2^-1]*L*xa+[delta_z1/2;delta_z2/2]*ones(1,Monte))...
    - floor([delta_z1^-1,0;0,delta_z2^-1]*[zrep1(1);zrep2(1)]+[delta_z1/2;delta_z2/2])*ones(1,Monte)    ...
       +ones(2,Monte)),[nz1;nz2]*ones(1,Monte)),ones(2,Monte))-nz2;
           
        ured= urep(Policy(indices));      
           disp(' made policy')
    % red model input
    % use Policy 
    u=R*ured+Q*xa+K*(x-P*xa);
    
     

x =A*x+B*u+Bw*w ;
xa =Aa*xa +Ba*ured+Bwa*w ;
 xun =A*xun+Bw*w;

Y(t+1,:)=C*x; 
Yun(t+1,:)=Ca*xa; 
%Yun(t+1,:)=C*xun;
%% Load new policy
savefile = ['Policy',num2str(t),'.mat'];  
load(savefile,'Policy');
    disp(num2str(t))
     

end

%sum(max(Yun(2:end,:)<=-.5+Epsilon|Yun(2:end,:)>=.5-Epsilon,[],1))/Monte
bound5=linspace(.5-Epsilon,.5,20);pr=[];pra=[];
for be=bound5
pr=[pr sum(max(Y(2:end,:)<=-be|Y(2:end,:)>=be,[],1))/Monte];
pra=[pra sum(max(Yun(1:end,:)<=-be|Yun(1:end,:)>=be,[],1))/Monte];
end

figure
plot(bound5,pr)
hold on
plot(bound5,pra,'g')
figure

  for i=1:Monte;
plot((1:7),Y(:,i),'x-b'); hold on;
  end

 ylim([-.6,.6])
saveas(gcf,'runmonte.png')

