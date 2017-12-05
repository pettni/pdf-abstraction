function [p, mu] = del_reach( MDP,NFA,del)
% del_reach computes a delta-reachability problem
% It takes as input:
%   1. MDP = a Markov decision process
%   2. Reach = a set of reach states or a NFA.
%   3. del =  the deviation in the probability
% It gives as output:
%   1. p = the robust probability
%   2. mu = the optimal robust policy


p =[];
mu =zeros(length(NFA.S),length(MDP.z_rep));


% - Stochastic transitions => MDP P

% - Antagonist  NFA.Trans

%% Value iteration : Initialise
% 
% V(z,q)=0 Initialize
% 
% 
%
V = zeros(length(NFA.S),length(MDP.z_rep));
V_new = zeros(length(NFA.S),length(MDP.z_rep));
V_aux=zeros(1,length(MDP.z_rep));
for k=120:-1:1
    
    % 1. For each q,z pair add the reach set 
    %   1_{F}(q) +  1_{Q\F}(q)* V
    for q_new = NFA.F
        V_new(q_new,:)=ones(length(MDP.z_rep),1);
    end
    if sum(abs(V_new(:)-V(:)))<eps
        disp('Break iteration, convergence reached')
        disp(['k=',num2str(k)])
    break
    end
        V=V_new;
    % 2. as a function of q_old pick the worst. But note, you can only pick
    % the worst from the allowed set of next q
    % remember :
    % Trans_new: DFA
    %.S_new x  MDP.y_rep x DFA.S_old => true/false
   
    for q_old = NFA.S
       % evaluate effect of nondeterminism
       V_aux= min(max(~NFA.Trans(:,:,q_old),V)); % make a V matrix with ones for not allowed transitions.
              %  min_(antagonist q_new)(1_{Q\q_old->qnew} + 1_{qold->q_new}
              %  V(x,q_new))
              %  
       % average over the transition distribution 
       V_new(q_old,:) = max( max(reshape(V_aux* MDP.P',length(MDP.z_rep),length(MDP.u_rep)),[],2)-del, 0) ;

       [~,index_mu(q_old,:)] = max(reshape(V_aux* MDP.P',length(MDP.z_rep),length(MDP.u_rep)),[],2) ;

      % we loose probability here!
    end
    
%     for q = NFA.S
%                figure(1)
%        subplot(length(NFA.S),1,q)
%        plot(MDP.z_rep,V_new(q,:))
%        title(['mode=',num2str(q)])
%        figure(2)
%        subplot(length(NFA.S),1,q)
%        plot(MDP.z_rep,index_mu(q,:))
%        title(['mode=',num2str(q)])
%     end
%     pause
    % check convergence!

end 
  % now its is time to compute the final value function. For this we look
  % at  initial NFA.S0
 for q_0 = NFA.S0
       % evaluate effect of nondeterminism
       p= min(max(~NFA.Trans(:,:,q_0),V));
 end
 
 % Compute the policy
for q_old = NFA.S
       % evaluate effect of nondeterminism
       V_aux= min(max(~NFA.Trans(:,:,q_old),V)); % make a V matrix with ones for not allowed transitions.
              %  min_(antagonist q_new)(1_{Q\q_old->qnew} + 1_{qold->q_new}
              %  V(x,q_new))
              %  
       % average over the transition distribution 
       [~,index_mu] = max(reshape(V_aux* MDP.P',length(MDP.z_rep),length(MDP.u_rep)),[],2) ;
       mu(q_old,:) = MDP.u_rep(index_mu);
      % we loose probability here!
end
end

