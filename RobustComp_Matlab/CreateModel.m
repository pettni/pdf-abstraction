% 1. start from continuous time model
dt=5*60; %sec
tend=400*60*60;
tend=floor(tend/dt)*dt;

Phis_Std=.1*10^3; %kW
% 2. weather model
desvar=1; % desired variance

% 3. init
x=[15;15];
ta=15;% mean outside%+ rand(0,1);



echo on
%% I. Model of house  ---------------------
% driven by:                               |
% 1. the heating (5kW range)               |
% 2. the outside weather                   |
% 3. the number of people inside           |
% 4. Combine ind. systems                  |
%-----------------------------------------
%% 1. Heating model of house 5kW
echo off
modelextsolar
echo on



tau=20*60; % time constant [sec] for the brownian weather noise.
sysweather=ss(-tau^-1,(1/2*tau)^-.5,1,[]);
echo off
var=covar(sysweather,1); % covariance of CT-model
sysweather.b=sysweather.b*(desvar/var)^.5; % correct covariance
sysweather.y='Tadel';
sysweather.u='genweather';
sysweather.StateName='weather_state';
%
sysweatherd=c2d(sysweather,dt);
var=covar(sysweatherd,1); % covariance of DT-model
sysweatherd.b=sysweatherd.b*(desvar/var)^.5;% correct covariance of DT-model
Sum = sumblk('Ta = Tadel+predict');
sysweatherd=connect(sysweatherd,Sum,{'genweather','predict'},{'Ta'});
sysweather=connect(sysweather,Sum,{'genweather','predict'},{'Ta'});

if allplots==1
    % test this
    genweather=randn(tend/dt+1,1); % input
    [Y,~,~]     = lsim(sysweather, [genweather/(dt^.5) 15*ones(size((0:dt:tend)))'],(0:dt:tend),'foh');
    [Yd,~,~]   = lsim(sysweatherd,[genweather 15*ones(size((0:dt:tend)))'],(0:dt:tend));
    figure;timeax=T/(60*60*24)+now;
    plot(timeax,Y);hold on;stairs(timeax,Yd)
    legend('CT-weather model','DT-weather model');title('weather simulated, 15 degrees pedicted');
    datetick2
end
echo on 

%% 3. Model of people in the building
%    100 W a person added to the building
%     When OCCUPIED
%          standard deviation 20W
%          mean when occupied 5*100W people
%   heat generated by people
n_people=10;
Q=100; % [W] heat generated by people
people_func=@(x) Q*(1*n_people+ sqrt(n_people)*.2*x);
echo off 

% B. When UNOCCUPIED
people_un=@(x) zeros(size(x));
echo on

%% 4. Combine systems %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   central heating (Q)     ------>| -------------- |
%   Human heat      (people) ----->|  House model   |---> (T_i) indoor T
%   Solar radiation (Phis)   ----->|
%   Coloured noise  (weather) ---->| ______________ |
%   variation on weather
% House model= sysdfull
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
echo off
sysdfull=connect(sysd,sysweatherd, ...
    {'Phih','people','Phis','genweather', 'predict'},{'Ti'});
sysfull=connect(sys,sysweather, ...
    {'Phih','people','Phis','genweather', 'predict'},{'Ti'}); % continuous model
%


if allplots==1
    Phih=3*10^3; % power of heating system
    tamean=10; % mean of ambient
 
    runs=20;
    
    one=ones(size((0:dt:tend)))';
    t=(0:dt:tend);
     Phis_t=.1*10^3+.1*10^3*randn(length(t),1);
    Yd=zeros(length(t),runs);
    Yd2=zeros(length(t),runs);
    % length of run:
    for i=1:runs
        people_occ    =  people_func(randn(size(t)));
        genweather  =randn(tend/dt+1,1); % input
        [Yd(:,i),~,~]   = lsim(sysdfull,[Phih*one,people_occ',Phis_t,...
            genweather, tamean*one],t,[15 15 15 15 0]);
        [Yd2(:,i),~,~]  = lsim(sysdfull,[Phih*one,people_occ',Phis_t,...
            genweather, 3*one],t,[15 15 15 15 0]);
    end
    
    figure('units','normalized','outerposition',[0 0 .5 .5])
    stairs(timeax,Yd,'Color',[.5,.5,.5]);hold on
    stairs(timeax,Yd2,'Color',[.75,.75,.75]);datetick2;
    title('Indoor air temperature')
    ylabel('temperature [^\circ C]')
    xlabel('Time')
    legend('Ambient 10 ^\circ C (mean)','Ambient 3 ^\circ C (mean)')
end


echo on

%% I--> Normalize system dynamics
%
sysdfull
%
% 1. Remove predict 
sysdfull=ss(sysdfull.a,sysdfull.b(:,1:end-1),sysdfull.c,sysdfull.d(:,end-1),dt);
%
% what we want
%n_people=10;
% ->  input range [0 and 1]
% ->  variance = identity
sysdfull.b= sysdfull.b*blkdiag(5*10^3,sqrt(n_people)*.2*100,Phis_Std,1);
% *  max power was 5kW, 
% *  std of people .2*100 (per person);
% *  std of solar energy
% *  variance of weather input was 1
%

