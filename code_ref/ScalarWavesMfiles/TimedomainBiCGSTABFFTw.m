clear all; clc; close all; clear workspace;
input = init(); 

    input = Wavelet(input);  
Wavelfrq = input.Wavelfrq;    
Wavelmax  = max(abs(Wavelfrq(:)));

% Redefine frequency-independent parameters
  input.N1 = 600;                            % number of samples in x_1             
  input.N2 = 600;                            % number of samples in x_2 
  input.dx = 1;                              % grid size
        x1 = -(input.N1+1)*input.dx/2 + (1:input.N1)*input.dx;   
        x2 = -(input.N2+1)*input.dx/2 + (1:input.N2)*input.dx;
  [input.X1,input.X2] = ndgrid(x1,x2);
  input.xS    = [0 ,-170];                   % source position
  input.c_sct = 3000;                         % wave speed in scatterer
  input       = initContrast(input);         % contrast distribution

Errcrr_0 = input.Errcri;

ufreq    = zeros(input.N1,input.N2,input.Nfft); 
for f = 2 : input.fsamples
    
%   (0) Make error criterion frequency dependent
        factor = abs(Wavelfrq(f))/Wavelmax;
        Errcri = min([Errcrr_0 / factor, 0.999]);     
        input.Errcri = Errcri;
        
%   (1) Redefine frequency-dependent parameters    
        freq = (f-1) * input.df;      disp(['freq sample: ', num2str(f)]);
        s = 1e-16 - 1i*2*pi*freq;          % LaPlace parameter
        input.gamma_0 = s/input.c_0;       % propagation coefficient
        input = initFFTGreen(input);       % compute FFT of Green function
       
%   (2) Compute incident field --------------------------------------------      
        u_inc = IncWave(input);

%   (3) Solve integral equation for contrast source with FFT --------------
        w =ITERBiCGSTABw(u_inc,input);     

%   (4) Compute total wave field on grid  and add to freqency components
        u_sct = Kop(w,input.FFTG);  
        ufreq(:,:,f) = Wavelfrq(f) .* (u_inc(:,:) + u_sct(:,:));
         
end; % frequency loop

ut = 2 * input.df * real(fft(ufreq,[],3));                    clear ufreq;
utime(:,:,1:input.fsamples) = ut(:,:,1:input.fsamples);       clear ut;

SnapshotU;                        % Make snapshots for a few time instants