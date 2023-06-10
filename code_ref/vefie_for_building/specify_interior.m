%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                                                                     %%%
%%%                       INTERIOR SPECIFICATION:                       %%%
%%%          position, phi, k, and rho                                  %%%
%%%          for each number of position (basis counter)                %%%
%%%                                                                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ct3 = 0;
ct4 = 0;
ct5 = 0;
basis_counter = 0 ; 
basis_wave_number(1:N*N,1) = k0;


for(ct1 = 1:M) % runs in y direction
    for(ct2 = 1:N) % runs in x direction
        basis_counter = (ct1-1)*N + ct2 ; % nr of position
        % ORIENTATION OF BASIS COUNTER WILL BE IN X DIRECTION!
        
        position(basis_counter,1) = start_pt + (ct2 - 0.5)*delta_x + j*(ct1 - 0.5)*delta_y  ; 
        temp_vec = position(basis_counter,1) - centre; 
        rho(basis_counter,1) = abs(temp_vec);
        the_phi(basis_counter,1) = atan2(imag(temp_vec),real(temp_vec));
        
        %%%%%%%%%%%%%%%%
        if building(ct1,ct2) == 1
            basis_wave_number(basis_counter) = k_d;
            ct3 = ct3 + 1;
            Concrete_configuration(ct3) = position(basis_counter,1);
        end
        if building(ct1,ct2) == 2
            basis_wave_number(basis_counter) = k_w;
            ct4 = ct4 + 1;
            Wood_configuration(ct4) = position(basis_counter,1);
        end
        if building(ct1,ct2) == 3
            basis_wave_number(basis_counter) = k_g;
            ct5 = ct5 + 1;
            Glass_configuration(ct5) = position(basis_counter,1);
        end
        %%%%%%%%%%%%%%%%%%%
    end 
end 

