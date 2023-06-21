function [E_inc,ZH_inc] = IncEMwave(input) 
gam0 = input.gamma_0;  
xS   = input.xS;    dx = input.dx;

% incident wave from electric dipole in negative x_1
  delta  = (pi)^(-1/2) * dx;       % radius circle with area of dx^2 
  factor = 2 * besseli(1,gam0*delta) / (gam0*delta);
  
  X1    = input.X1-xS(1);      X2 = input.X2-xS(2); 
  DIS   = sqrt(X1.^2 + X2.^2); 
  X1    = X1./DIS;             X2 = X2./DIS;
    G   =   factor *         1/(2*pi).* besselk(0,gam0*DIS);  
   dG   = - factor * gam0 .* 1/(2*pi).* besselk(1,gam0*DIS);
   dG11 = (2*X1.*X1 - 1) .* (-dG./DIS) + gam0^2 * X1.*X1 .* G;
   dG21 =  2*X2.*X1      .* (-dG./DIS) + gam0^2 * X2.*X1 .* G;
   
  E_inc{1} = - (-gam0^2 * G + dG11);  
  E_inc{2} = - dG21;    
  E_inc{3} = 0; 
  
 ZH_inc{1} = 0; 
 ZH_inc{2} = 0;
 ZH_inc{3} = gam0 * X2 .* dG; 