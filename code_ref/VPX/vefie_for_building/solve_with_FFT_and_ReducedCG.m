%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Reduced CG algoritm with FFT, solving Z*E=V with Z =I+GD %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear r p error icnt a b

Rfo = logical(D); % Reduced foreward operatior
Vred = Rfo.*V; % create V reduced
Ered = zeros(basis_counter, 1); % gues

r = Rfo.*Ered + Rfo.*BMT_FFT(G_vector.',D.*Ered,N) - Vred; % Z*E - V (= error)
p = -(Rfo.*r + conj(D).*(BMT_FFT(conj(G_vector.'),Rfo.*r,N))); % -Z'*r

tol = 1e-3; % error tolerance
icnt = 0; % iteration counter
error = abs(r'*r); % definition of error

fprintf('\nStart reduced CG iteration with 2D FFT\n')
start3 = tic;

while ( error > tol )&&( icnt <= basis_counter )
  icnt = icnt+1;
  
  if( mod(icnt,50) == 0 ) 
        fprintf(1,'%dth iteration Red \n',icnt) ; 
  end
  a = (norm(Rfo.*r + conj(D).*BMT_FFT(conj(G_vector.'),Rfo.*r,N))/norm(Rfo.*p + Rfo.*(BMT_FFT(G_vector.', D.*p,N))))^2; %(norm(Z'*r)^2)/(norm(Z*p)^2);
  Ered = Ered + a*p;
  r_old = r;
  r = r + a*(Rfo.*p + Rfo.*(BMT_FFT(G_vector.',D.*p,N))); % r = r + a*z*p
  b = (norm(Rfo.*r + conj(D).*BMT_FFT(conj(G_vector.'),Rfo.*r,N))/norm(Rfo.*r_old + conj(D).*BMT_FFT(conj(G_vector.'),Rfo.*r_old,N)))^2; %b = (norm(Z'*r)^2) /(norm(Z'*r_old)^2);
  p = -(Rfo.*r + conj(D).*BMT_FFT(conj(G_vector.'),Rfo.*r,N)) + b*p; % p=-Z'*r+b*p
  
  error = abs(r'*r);
  Reduced_iteration_error(icnt,1)= abs(r'*r);
end
   
time_Red = toc(start3);
