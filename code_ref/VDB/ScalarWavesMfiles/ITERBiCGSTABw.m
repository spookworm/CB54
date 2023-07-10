function [w] = ITERBiCGSTABw(u_inc,input)
% BiCGSTAB_FFT scheme for contrast source integral equation Aw = b
itmax  = 1000;  
Errcri = input.Errcri;    

b = input.CHI(:) .* u_inc(:);                           % known 1D vector

if exist(fullfile(cd, 'w_P.mat'), 'file')
    temp = importdata('w_P.mat');
    x0 = temp(:);
    clear temp;
else
    x0 = zeros(size(b));
end
% w = bicgstab(@(w) Aw(w,input), b, Errcri, itmax);       % call BiCGSTAB
[w, flag, relres, iter, resvec] = bicgstab(@(w) Aw(w, input), b, Errcri, itmax, [], [], x0);

% display(flag)
display(relres) % Final
display(iter)
% display(resvec)
display(vpa(resvec, 20));
w = vector2matrix(w,input);                             % output matrix

end %----------------------------------------------------------------------

function y = Aw(w,input)
  w = vector2matrix(w,input);               % Convert 1D vector to matrix  
  y = w -  input.CHI .* Kop(w,input.FFTG);
  y = y(:);                                 % Convert matrix to 1D vector
end %----------------------------------------------------------------------

function w = vector2matrix(w,input)
% Modify vector output from 'bicgstab' to matrix for further computations
global nDIM;
  if nDIM == 2
      w = reshape(w,[input.N1,input.N2]);
  elseif nDIM == 3
      w = reshape(w,[input.N1,input.N2,input.N3]);
  end
end