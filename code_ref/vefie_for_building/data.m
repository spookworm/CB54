%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Defined general variables, and variables in/oud side the scatterer %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%% general %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
MHz = 1000000.0 ;
f = 60.0*MHz ; 
omega = 2.0*pi*f ; 

%%% Oudside scatter
epsilon0 = 8.854e-12 ; % in case of vacuum
mu0 = 4.0*pi*1.0e-7 ; % in case of vacuum
c = 1.0/sqrt(epsilon0*mu0) ; 
k0 =  omega*sqrt(mu0*epsilon0) ;
eta0 =  sqrt(mu0/epsilon0) ;
lambda0 = c / f ; 

%%% Inside scatterrer %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%<------BIGGEST RELATIVE EPSELON HAS TO BE CALLED AS FOLLOW
% using concrete 
epsilonr = 6.0 -(0.0*j); 
epsilon_d = epsilonr*epsilon0 ; 
mu_d = mu0 ; % No magnetic permitivity in wood
c_d = 1.0/sqrt(epsilon_d*mu_d) ; 
k_d = omega*sqrt(mu_d*epsilon_d) ;
eta_d =  sqrt(mu_d/epsilon_d) ;
lambda_d = c_d / f ;

%<------- OTHER MATERIALS
% using Glass
epsilonrg = 4.0 - (0.0*j);
epsilon_g= epsilonrg*epsilon0 ; 
mu_g = mu0 ; % No magnetic permitivity in wood.
c_g = 1.0/sqrt(epsilon_g*mu_g) ; 
k_g = omega*sqrt(mu_g*epsilon_g) ;
eta_g =  sqrt(mu_g/epsilon_g) ;
lambda_g = c_g / f ;

% using wood
epsilonrw = 2.0 -(0.0*j); 
epsilon_w= epsilonrw*epsilon0 ; 
mu_w = mu0 ; % No magnetic permitivity in wood.
c_w = 1.0/sqrt(epsilon_w*mu_w) ; 
k_w = omega*sqrt(mu_w*epsilon_w) ;
eta_w =  sqrt(mu_w/epsilon_w) ;
lambda_w = c_w / f ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
