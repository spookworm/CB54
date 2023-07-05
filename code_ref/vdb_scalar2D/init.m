function input = init()
% Time factor = exp(-iwt)
% Spatial units is in m
% Source wavelet  Q = 1

input.c_0 = 3e8; % wave speed in embedding
input.c_sct = input.c_0*2.75; % wave speed in scatterer


f = 10e6; % temporal frequency
wavelength = input.c_0 / f; % wavelength
s = 1e-16 - 1i * 2 * pi * f; % LaPlace parameter
input.gamma_0 = s / input.c_0; % propagation coefficient
disp(['wavelength = ', num2str(wavelength)]);

% add input data to structure array 'input'
input = initSourceReceiver(input); % add location of source/receiver

input = initGrid(input); % add grid in either 1D, 2D or 3D

input = initFFTGreen(input); % compute FFT of Green function

input = initContrast(input); % add contrast distribution

input.Errcri = 1e-15;

end % function