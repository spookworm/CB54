%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Use of the function to compute the scattered field at each receivers ,
% for each source
% Following by the implementation of the CSI Algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Created : 08/2013
% Author: Elodie Bolus

clc;
close all;
clear all;

%% Initialize general variables
tic;
fprintf('Initialization of the process \n');

% Frequency
MHz = 1000000.0;
f = 500.0*MHz;
w_puls = 2*pi*f;

% Conductivity and permittivity definitions
eps_0 = 8.854e-12;
mu_0 = 4.0*pi*1.0e-7;
k_0 = w_puls*sqrt(eps_0*mu_0);
c = 1/(sqrt(eps_0*mu_0));

receivers = 3; % Numbers of sources/receivers
N_dis = 29; % Number of subspaces for the imaging domain D (D=N_dis*N_dis)
TOL = 10^-6; % Tolerance error

radius_S = 3*(c/f)/(2*pi) ; % Radius of the measurement surface S

%% Computation of the incident field (in the imaging domain) and the scattered field (on the measurement surface ) for each source
fprintf ('Star of the computation for each source \n');
tic;

f_sct = zeros(receivers, receivers); % Scattered field initialization
u_inc = zeros(N_dis*N_dis, receivers); % Incident field initialization

for cnt=1:receivers
	fprintf('Source number: %d \n', cnt);
	% Position of source , the center of the measurement surface is [0 ,0]
	
	S_position = [radius_S*cos(2*pi/receivers*cnt), radius_S*sin(2*pi/receivers*cnt)];

	% Compute the fields and keep the information
	[E_sct, E_inc, LeafCube_sl, LeafCube_center, H0, khi_exact] = Compute_fields(eps_0, mu_0, f, N_dis, receivers, S_position);

	% Save scattered field and incident field for the source numbered cnt
	f_sct(:, cnt) = E_sct;
	u_inc(:, cnt) = E_inc;
end

toc;
fprintf('End of the computation for each source \n');

%% Variables for CSI algorithm
itr_max = 512; % Iteration maximum of the algorithm if it doesn ✬ t converge
itr = 1; % Starting value for the algorithm
area_cube = LeafCube_sl.^2; % Area of each cube

eta_d = 0; % Will depend on the iteration number
eta_s = 1/sum(sum(abs(f_sct).^2)); % Do not depend on the iteration

%% Position receivers and calculation of distances
receiver_position =zeros(2, receivers);
% Compute all receivers coordinates
for cnt = 1:receivers
	receiver_position(:, cnt) = [radius_S*cos(2*pi/receivers*cnt); radius_S*sin(2*pi/receivers*cnt)];
end

% Calculation of the distance between each receiver and each point of the domain imaging
Obj_dis = zeros(N_dis*N_dis, receivers);
for cnt =1:receivers
	% Computation of the vector between each each receiver and each point of the domain
	vector_domain = LeafCube_center - repmat(receiver_position(:, cnt).', [size(LeafCube_center, 1) 1]);
	Obj_dis(:, cnt) = sqrt(vector_domain(:, 1).^2 + vector_domain(:, 2).^2); % Computation of the distance
end

% Calculation of the distance between each point of the domain imaging
% Obj dis d = zeros (N dis✯ N dis , N dis✯ N dis) ;
Obj_dis_d = repmat(LeafCube_center(1, :), [N_dis*N_dis 1]) - LeafCube_center;
Obj_dis_d = sqrt(Obj_dis_d(:, 1).^2+Obj_dis_d(:, 2).^2);

%% Definition of the different operators
% Gs operator
distance = Obj_dis.';
Gs = k_0.^2*(1/(4*1i))*besselh(0, 2, k_0*distance)*area_cube;

% Gs star operator mapping L2(S) into L2(D)
Gs_star = (conj(k_0.^2))*conj((1/(4*1i))*besselh(0, 2, k_0*Obj_dis))*area_cube;

% Gd operator
Gd = (k_0.^2*(1/(4*1i)))*besselh(0, 2, k_0*Obj_dis_d)*area_cube;
% Diagonals terms are undefined ie for besselh(0, 2, 0). We calculate them
% with relation in the book ”Computation Methods for Electromagnetics ”. See
% equations (2.69) and (2.74) . H0 is calculated in
% ”Compute_fields.m”
Gd(1) = (k_0.^2*(1/(4*1i)))*H0;

% Gd star operator mapping L2(D) into L2(D)
Gd_star = conj(k_0.^2)*conj((1/(4*1i))*besselh(0, 2, k_0*Obj_dis_d))*area_cube;
% As above we have to compute the diagonal term separately
Gd_star(1) = conj(k_0.^2)*conj((1/(4*1i))*H0);

%% Starting values
fprintf ('Initialization of the value. Iteration 0 \n');

% Contrast source values for the iteration 0
w = zeros(N_dis*N_dis, receivers); % Initialization of the state equation
A = (sum(sum(abs(Gs_star*f_sct).^2))/sum(sum(abs(Gs*Gs_star*f_sct).^2)));
B = Gs_star*f_sct;
w = A.*B;

% Total field value for the iteration 0
u = zeros (N_dis*N_dis, receivers); % Initialization of the total field in D
for cnt=1:receivers
	u (:, cnt) = u_inc(:, cnt) + BMT_FFT(Gd, w(:, cnt)); % Gd and Gd star are computed using the FFT routines
end

% Calculation of the contrast value for the iteration 0
sum_w_u = sum(w.*conj(u), 2);
sum_u = sum(abs(u).^2, 2);
% khi = zeros(N_dis*N_dis, 1); % Contrast
khi = sum_w_u./sum_u; % Contrast value for the iteration 0

%% Data and states errors
%rho = zeros ( receivers , receivers ) ; % Data error
rho = f_sct - Gs*w; % Data error
r = zeros(N_dis*N_dis, receivers); % Object error
for cnt=1:receivers
	r(:, cnt) = khi.*u_inc(:, cnt) - w(:, cnt) + khi.*BMT_FFT(Gd, w(:, cnt)); % Object error
end

%% Initialization values for the Algorithm
fprintf('Beginning of CSI algorithm \n');
v = zeros(N_dis*N_dis, receivers); % Update direction for the contrast source (v = 0 at iteration 0)
grad_w = zeros(N_dis*N_dis, receivers); % Gradient of the cost functional

% Initialization of the cost functional
sum_rho = sum(sum(abs(rho).^2));

costfunctional = eta_s*sum_rho; % eta d only exist for n > 1. We initialize the cost functional by F = eta_s*rho
cost_prev = costfunctional;
error = norm(khi_exact - khi);

%% While loop of the CSI algorithm
% while ( itr < 513) && ( costfunctional > TOL) % could be another solution
% of the statement for the while loop
while(itr < 513) && ( error > TOL )
	fprintf('Iteration %d \n', itr);

	% First Goal: Update of the contrast source
	% 1. Calculation of eta d changing for every iteration
	for cnt=1:receivers
		eta_d = eta_d + norm(khi.*u_inc(:, cnt)).^2;
	end
	eta_d = 1/eta_d;
	
	% 2. Calculation of the gradient of the cost functional
	grad_w_prev = grad_w ; % Save the previous value of grad w (at iteration n-1)
	for cnt=1:receivers
		grad_w(:, cnt) = -eta_s*Gs_star*rho(:, cnt) - eta_d*(r(:, cnt) - BMT_FFT(Gd_star, conj(khi).*r(:, cnt))); % New value of grad w
	end

	% 3. Update directions with Polak-Ribiere CG directions
	v_prev = v ; % Save the previous value of v (at iteration n-1)
	sum_num = sum(sum(grad_w.*(grad_w-grad_w_prev)));
	sum_denum = sum(sum(grad_w_prev.*grad_w_prev));
	if (itr==1) % For the first iteration v = grad w
		v = grad_w;
	else
	v = grad_w + (real(sum_num)/sum_denum)*v_prev; % Update of the direction for itr > 2
	end

	% 4. Update of the contrast source
	sum_denum_1 = sum(sum(abs(Gs*v).^2));
	sum_denum_2 = 0;
	for cnt=1:receivers
		sum_denum_2 = sum_denum_2 + (norm(v(:, cnt) - khi.*BMT_FFT(Gd, v(:, cnt)))).^2;
	end
	
	sum_num_alpha = sum(sum(grad_w.*v));
	% Upadte of the real contrast parameter alpha w
	alpha_w = (-real(sum_num_alpha))/(eta_s*sum_denum_1 + eta_d*sum_denum_2);
	w = w + alpha_w*v; % Update of the contrast source
	rho = f_sct - Gs*w; % Update of the data error
	% Second Goal : Update of the contrast
	% 1. Update of the total field in D
	
	for cnt=1:receivers
		u(:, cnt) = u_inc(:, cnt) + BMT_FFT(Gd, w(:, cnt));
	end

	% 2. Update of the preconditioned gradient (update directions)
	sum_num_d = zeros(N_dis*N_dis, 1);
	for cnt =1:receivers
		sum_num_d = sum_num_d + (w(:, cnt) - khi.*u(:, cnt)).*conj(u(:, cnt));
	end

	sum_denum_d = sum((abs(u).^2), 2);
	d = eta_d*(sum_num_d./sum_denum_d); % Update of the preconditioned gradient

	% 3. Update of the contrast
	alpha_khi = 1/eta_d; % Real contrast parameter
	khi_prev = khi;
	khi = khi_prev + alpha_khi*d; % Update of the contrast
	% 3. bis Addition of the positivity constraint after the upadte of the contrast
	Real_khi = real(khi);
	Imag_khi = imag(khi);

	for cnt=1:N_dis*N_dis
		if (Real_khi(cnt) < 0)
			Real_khi(cnt) = 0;
		else Real_khi(cnt) = Real_khi(cnt);
		end
		if (Imag_khi (cnt) < 0)
			Imag_khi (cnt) =0;
		else Imag_khi(cnt) = Imag_khi(cnt);
		end
	end
	
	khi = Real_khi + 1i*Imag_khi;

	% 4. Update of the Object error
	for cnt=1:receivers
		r(:, cnt) = khi.*u_inc(:, cnt) - w(:, cnt) + khi.*BMT_FFT(Gd, w(:, cnt)); % Object error
	end

	% Update of the cost functional
	sum_rho = sum(sum(abs(rho ).^2));
	sum_r = sum(sum(abs(r).^2));
	costfunctional = eta_s*sum_rho + eta_d*sum_r;
	% error tot(itr) = costfunctional;
	% fprintf('Relative Error norm of iteration %d is : %5.3f \n', itr, error_tot(itr));
	error = norm(khi_exact - khi);
	error_tot(itr) = error;
	fprintf('Relative Error norm of iteration %d is : %5.3f \n', itr, error_tot(itr));
	data_error(itr) = sum_rho; % Save the data error at each iteration
	object_error (itr) = sum_r; % Save the object error at each iteration
	itr = itr + 1;
end

%% Plot of results
% Plot errors
figure(1);
semilogy(error_tot, '-x', 'color', 'red');
hold on
xlabel('Number of iterations');
ylabel('Errors');
title('CSI Algorithm');
semilogy(data_error, '-x', 'color', 'blue');
semilogy(object_error, '-x', 'color', 'green');
legend('Relative error', 'Data Error', 'Object Error');
hold off

% Create a graph of the real part of the object
X = zeros(1, N_dis);
for cnt6=1:N_dis
	X(:, cnt6) = cnt6;
end

Y = zeros(1, N_dis);
for cnt6=1:N_dis
	Y(:, cnt6 ) = cnt6;
end

Z = zeros(N_dis, N_dis);
for cnt6=1:N_dis
	for cnt7=1:N_dis
		Z(cnt6, cnt7) = real(khi_exact((cnt6 -1)*N_dis+cnt7));
	end
end

% Plot the real part of the object expected
figure(2);
surf(X, Y, Z);
hold on;
zlabel('Real part of khi');
title('Real part of the object');
hold off;

% Create a graph of the real part of the object obtained
Z = zeros(N_dis, N_dis);
for cnt6=1:N_dis
	for cnt7=1:N_dis
	Z(cnt6, cnt7) = real(khi((cnt6 -1)*N_dis+cnt7));
	end
end

% Plot the real part obtained
figure(3);
surf(X, Y, Z);
hold on;
zlabel('Real part of khi');
title('Real part of the object found by the CSI algorithm');
hold off;

% Create a graph of the imaginary part of the object
Z = zeros(N_dis, N_dis);
for cnt6=1:N_dis
	for cnt7=1:N_dis
		Z(cnt6, cnt7) = imag(khi_exact((cnt6-1)*N_dis+cnt7));
	end
end

% Plot the imaginary part of the object expected
figure(4);
surf(X, Y, Z);
hold on;
zlabel('Imaginary part of khi');
title('Imaginary part of the object');
hold off;

% Create a graph of the iaginary part of the object
Z = zeros(N_dis, N_dis);
for cnt6=1:N_dis
	for cnt7=1:N_dis
		Z(cnt6, cnt7) = imag(khi((cnt6-1)*N_dis+cnt7));
	end
end

% Plot the imaginary part of the object obtained
figure(5);
surf(X, Y, Z);
hold on;
zlabel('Imaginary part of khi');
title('Imaginary part of the object found by the CSI algorithm');
hold off;

toc;