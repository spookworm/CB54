%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                                                                     %%%
%%%  CREATION OF ELEMENTS FOR VEFIE: (I+GD)E=V                          %%%
%%%  (Volume-equivalent Electric Field Intergral Equation)              %%%    
%%%                                                                     %%%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

V = zeros(basis_counter,1); % incident field
D = zeros(basis_counter,1); % diagonal contrast matrix stored as a vector
G_vector = zeros(basis_counter,1); % BMT dense matrix, stored as a vector
    
fprintf('\nStart creation of all %d ellements of G,V and D \n \n', basis_counter)
start1 = tic;

for(ct1 = 1:basis_counter) 
    
    if( mod(ct1,100) == 0 ) 
        fprintf(1,'%dth ellement \n',ct1) ; 
    end
    V(ct1) =  exp(-j*k0*rho(ct1)*cos(the_phi(ct1))) ; % incident field
    D(ct1,1)= (basis_wave_number(ct1,1)*basis_wave_number(ct1,1) - k0*k0); % contrast function                          
    R_mn2 = abs(position(ct1) -position(1)) ;     
 
    if ct1 == 1
        G_vector(ct1,1) = (j/4.0)*((2.0*pi*equiv_a/k0)*(besselj(1,k0*equiv_a) - j*bessely(1,k0*equiv_a))- 4.0*j/(k0*k0) ) ;
    else
        G_vector(ct1,1) = (j/4.0)*(2.0*pi*equiv_a/k0)*besselj(1,k0*equiv_a)*(besselj(0,k0*R_mn2) - j*bessely(0,k0*R_mn2)) ;
    end
  
end
Time_creation_all_elements_from_G_V_and_D = toc(start1)
    
