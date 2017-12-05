function [ eps,del,Q,R,P,M ] = epsdel_compute(sysfull,sys, M,K,rad,del)
% BISIMAP
% Approximate bisim
% Given
%   sysfull = full order model
%   sys     = low order mode
%   m       = number of controllable input (rest is presumed white noise inputs)
%   M       = initial M;
n=length(sysfull.a);
ns=length(sys.A);

% assume no loss of controls
m=1; % same number for both
mw=size(sysfull.b,2)-m;

A  =sysfull.a;
B  =sysfull.b(:,1:m);
Bw =sysfull.b(:,m+1:end);
C  =sysfull.c;

As  =sys.A;
Bs  =sys.B;
Bws =sys.Bw;
Cs  =sys.C;

%% 1. Compute Sylverster
cvx_begin SDP quiet 
        variable P(n,ns)
        variable Q(m,ns)
        variable R(m,m)
        variable lm semidefinite
        variable lw semidefinite
    minimize(lw+2*lm)
        subject to
        zeros(n,ns)==A*P+B*Q-B*K*P-P*As; 
        Cs==C*P;
        0<=[M M*(B*R-P*Bs);(B*R-P*Bs)'*M lm*eye(m)];
        0<= [M M*(Bw-P*Bws);(Bw-P*Bws)'*M lw*eye(mw)];

cvx_end
    Q=Q-K*P;
    display(['Q = ',num2str(Q)])
% %% Optimise K
% 
% % 1. First try  
% 
% 
% %X-(A+BK)X(A+BK)'-cw*(Bw-P*Bws)(Bw-P*Bws)>=0
% % xx'<=X
% % Kxx'K'<=KXK'<=.3*e(or something else)
% % .3e-KXK'>=0
% % [.3 K;  
% %   K' X^-1]
% 
% % X-[(A+BK)X (Bw-P*Bws)] [X^-1 0;0 cwI][(A+BK)X (Bw-P*Bws)]'>=0
% %[X      0       | X(A+BK)'       ]
% %[ 0        cw^-1I  |  (Bw-P*Bws)'  ]
% %[................. |  X            ]>=0
% 
% % ( define based on X and L=KX)
% Noise=sqrt((Bw-P*Bws)*(Bw-P*Bws)'*16+(B*R-P*Bs)*(B*R-P*Bs)'*.05);
% cvx_begin SDP quiet
%             variable X(n,n) semidefinite
%             variable L(1,n)   
%             variable t2 semidefinite
%         minimise(t2)
%         subject to 
%             0<=[5 L;L' X];
%             C*X*C'<=.2
%             0    <=[    X,            zeros(n,n),     X*A'+ L'*B';...
%                         zeros(n,n),   t2.*eye(n),     Noise';...
%                         A*X+B*L,      Noise ,          X];
%         
% cvx_end
% K=L/X;
% K1=K;
% display(['First try K =', num2str(K1)]);
K=-K
%% Compute Delta vs Epsilon
  max(abs(eig(A)))
  
  deltarange=[1-logspace(log10(1),-3,10)];

  if exist('del')
      deltarange=[1-del];

  end 


precision=10^-10;
increment=(1-max(abs(eig(A+B*K))))/10;
t1=min( .2+.8 *max(abs(eig(A))),1-10^-9); 
if  ~exist('it_max','var')
    it_max=50;
end
eps=[]; 
i=1;pk=[];
AB=[A+B*K Bw-P*Bws B*R-P*Bs P]
cu=max(sys.U.V)^2

noise_acc = covar(ss(A+B*K,Bw-P*Bws,eye(3),[],1),1)
input_acc = covar(ss(A+B*K,B*R-P*Bs,eye(3),[],1),cu)
rad_acc = covar(ss(A+B*K,P,eye(3),[],1),rad^2)
cvx_solver sedumi
for Delt_comp=deltarange;%1-logspace(log(.4),log(.1),10) %1-\delta
    fprintf(['(1-delta=',num2str(Delt_comp),'),  '])
    it=1;conte=1; 
    t1range=[];

    cw=chi2inv(Delt_comp,mw)
    
    while conte==1
        t1range=[t1range t1];
    
 
        cvx_begin SDP quiet
        
        variable M(n,n) semidefinite
        variable t2 semidefinite
        variable t3 semidefinite
        variable t4 semidefinite
        variable tinvd semidefinite
        
        minimise((t2*cw+cu*t3+rad^2*t4))
        subject to
        0<= [M C';C 1];
        AB'*M*AB ...
            -[eye(n)*t1 zeros(n,m+mw+ns)]'*M*[eye(n)*t1 zeros(n,m+mw+ns)]...
            -[zeros(mw,n) eye(mw) zeros(mw,m+ns)]'*t2*eye(mw)*[zeros(mw,n) eye(mw) zeros(mw,m+ns)]...
            -[zeros(m,n+mw) eye(m) zeros(m,ns)]'*t3*[zeros(m,n+mw) eye(m) zeros(m,ns)]...
            -[zeros(ns,n+mw+m) eye(ns)]'*t4*[zeros(ns,n+mw+m) eye(ns)]<= 0
        
        cvx_end
        
        if (length(cvx_status)~=length('Solved')) || (cvx_status(1)~='S')
            t1=min(t1+abs(increment)*.1,1);
            fprintf(cvx_status)
        else
            pk=[pk,-sqrt((cw*t2+cu*t3+rad^2*t4)/(1-t1^2)) ];

                       
            if length(pk) >1
                if (pk(end)-pk(end-1) )<=0
                    increment=increment*(-.44);
                    t1=min(t1+increment,1-precision);
                else
                    t1=min(t1+increment,1-precision);
                end
            else
                t1=min(t1+increment/10,1-precision);
            end
            
        end
        it=1+it;
        if it>it_max
            conte=0;
            t1=t1range(end);
        end
        if length(pk)>3 % after 3 iterations start checking whether optimility is below precision
            if (abs(pk(end)-pk(end-1))<precision) && (abs(pk(end-1)-pk(end-2))<precision)
                conte=0;
                t1=t1range(end);
            end
        end 
 
    end
    fprintf(['\n eps=',num2str(-max(pk)),', ']) ;fprintf(['t1=',num2str(t1),', '])
     fprintf([' it=',num2str(it),', '])
   disp((pk))
    disp(' ')
   % disp('end linesearch')
    increment=max(abs(t1range(1)-t1range(end))/3,10^-4);

    eps=[eps -max(pk)]; 
    
    pk=[];
end
%figlast=figure;
plot(1-deltarange,eps);
xlabel('\delta')
ylabel('\epsilon')
evalin('caller','figrem=figure(gcf);');
disp('<a href="matlab: figure(figrem)">Figure</a> of delta vs epsilon ');

del=1-deltarange
end

