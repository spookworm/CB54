clear all; clc; close all; clear workspace;
input = initEM(); 
  
%  (1) Compute analytically scattered field data --------------------------
EMForwardCanonicalObjects
       
plotEMcontrast(input); % plot permittivity / permeability contrast
   
%  (2) Compute incident field ---------------------------------------------      
[E_inc, ZH_inc] = IncEMwave(input);
% E_inc1=E_inc{1};
% E_inc2=E_inc{2};
% E_inc3=E_inc{3};

%  (3) Solve integral equation for contrast source with FFT ---------------
tic;
[w_E] = ITERBiCGSTABwE(E_inc,input);
toc;
w_E1=w_E{1};
w_E2=w_E{2};

plotContrastSourcewE(w_E,input);
[E_sct] = KopE(w_E,input); 
plotEtotalwavefield(E_inc,E_sct,input)

%  (4) Compute synthetic data and plot fields and data --------------------
[Edata,Hdata] = DOPwE(w_E,input);             
displayEdata(Edata,input);
displayHdata(Hdata,input);