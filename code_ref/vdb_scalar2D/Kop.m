function [Kv] = Kop(v, FFTG)
% disp(size(v))
Cv = zeros(size(FFTG)); % make fft grid
[N1, N2] = size(v);
Cv(1:N1, 1:N2) = v;

Cv = fftn(Cv);
Cv = ifftn(FFTG.*Cv); % convolution by fft

Kv = Cv(1:N1, 1:N2);