function NFA_result=  NFA_eps(DFA,eps,MDP,K)
%NFA_EPS computes the nodeterministic NFA based on a given DFA, an epsilon
%error and Polytopes representing atomic propositions

eps
% This is the composition of the given MDP
% MDP.y_rep; ;
% we only care about y_rep.
%
% DFA.S= [1,2]; % set of states
% DFA.S0 = [1]; % initial state
% DFA.Act = ['k','nk'];
% DFA.F = [2]; % target states
% DFA.Trans=[2 1; 2 2]; % column = input (k,nk) , row = state q, value = next state
% % https://spot.lrde.epita.fr/trans.html

% extend and shrink the Polytope

% 1. compute K and not K for all y
K_max= K+Polyhedron([-eps,eps]'); % max size of  K 
K_min= K-Polyhedron([-eps,eps]'); % min size of K
ynK=~K_min.contains(MDP.y_rep(:,:));
yK=K_max.contains(MDP.y_rep(:,:));

Trans_new=zeros(length(DFA.S),length(MDP.y_rep),length(DFA.S));
for q_old=DFA.S % 
    Trans_new(DFA.Trans(q_old,1),yK,q_old)=ones(1,sum(yK)); % => the index of a potential new Q 
    Trans_new(DFA.Trans(q_old,2),ynK,q_old)=ones(1,sum(ynK)); % => the index of a potential new Q 

end

NFA_result = [];
NFA_result.Trans=Trans_new;
NFA_result.act=MDP.y_rep;
NFA_result.F=DFA.F;
NFA_result.S0 = DFA.S0; % initial state
NFA_result.S = DFA.S; % initial state

end

