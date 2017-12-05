%% The model Ti TM Te  Ria
% based on: Bacher, P., Madsen, H.:
% Identifying suitable models for the heat dynamics of buildings.
% Energy Build. 43, 7, 1511?1522 (2011).
%
% dt= Sampling time for the discretisation
%
% Temperatures in the building
% Ts= sensor T.
% Ti= interior T (i.e., indoor air)
% Tm= interior thermal medium (walls and furniture)
% Th= heaters T
% Te= building envelope T
%---------------------------------------------------------------------


%% The parameters

% 1. Heat capacity [kwh/C] of
Ci=  .143   *1000*60^2;% interior 
Ce=  3.24    *1000*60^2;% building envelope
Ch=  .321    *1000*60^2;% electrical heaters
Cs=  .619  *1000*60^2;% temp. sensor

% 2. The thermal resistance [C/kw] of
Rie=  .909  *10^-3;% interior and building envelope 
Rea=  4.47  *10^-3;% building envelope and ambient
Rih=  .383  *10^-3;% int-sensor;
Ris=  .115  *10^-3;% heater and the interior


% 3. areas [m2]
Aw= 6.03   ;% effective window area 
Ae=0;


%% deterministic dynamics 
% driven by heater with capacity up to 5 kW and by the ambitent temperature
% The equations

syms Ts Ti Ta Th Te 
syms Phih
syms people
syms Phis

% The sensor equation
dTs=(Ris*Cs)^-1*(Ti-Ts);
dTi=(Ris*Cs)^-1*(Ts-Ti)+(Rih*Ci)^-1*(Th-Ti)...
                + (Rie*Ci)^-1*(Te-Ti)+ Ci^-1*people+ Ci^-1*Aw*Phis;
dTh=(Rih*Ch)^-1*(Ti-Th)+(Ch)^-1*Phih;
dTe=(Rie*Ce)^-1*(Ti-Te)+(Rea*Ce)^-1*(Ta-Te)+ Ce^-1*Ae*Phis;

A=double(jacobian([dTs;dTi;dTh;dTe],[Ts, Ti, Th, Te]));
Bt=double(jacobian([dTs;dTi;dTh;dTe],[Ta]));
Bh=double(jacobian([dTs;dTi;dTh;dTe],Phih));
Bs=double(jacobian([dTs;dTi;dTh;dTe],Phis));
Bpeople=double(jacobian([dTs;dTi;dTh;dTe],people));
sys=ss(A,[Bh Bt Bpeople,Bs],[0 1 0 0],[]);
sys.StateName={'Ts', 'Ti', 'Th', 'Te'};
sys.u={'Phih','Ta','people','Phis'};
sys.y={'Ti'};
sysd=c2d(sys,dt);
 


