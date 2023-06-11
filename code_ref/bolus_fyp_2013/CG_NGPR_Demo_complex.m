%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nonlinear Conjugate Gradient Method with Secant and Polak−Ribiere with
% complex values
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author : Elodie Bolus
% Created : 06/2013
clc
clear all
close all

% Definition of the matrix A, the vector b and their sizes
size_x = 500;
A = generateSPDmatrix(size_x) + i*generateSPDmatrix(size_x);
b = rand(size_x, 1) + i*rand(size_x, 1);
%% Solve the equation using matrix inverse
fprintf('Value of x without Conjugate Gradient Method \n');
x = A\b

fprintf('\n', x);
%% Solve the equation using Nonlinear Conjugate Gradient with Secant and PR
% Initialization of the solution vector x
x_CG = zeros(size_x, 1);

% Initialization of the algorithm
itr = 2; % Iteration number
k = 0; % Restart parameter

f_deriv = 1/2*A.'*x_CG(:, 1) + 1/2*A*x_CG(:, 1) - b;
r = -f_deriv;
u = diag(A);
M = diag(u); % Preconditioner value

s = M\r;
d = s;

delta_new = abs(r.'*d);
delta_zero = delta_new;

% Defintion of the maximum values for iterations
itr_max = 2*size_x;
jtr_max = size_x;

sigma_0 = 0.01; % Step parameter
TOL = 10^-6; % Error tolerance

while (itr < itr_max) && (delta_new > delta_zero*TOL)
	jtr = 0;
	delta_d = d.'*d;
	
	alpha_cg = -sigma_0;
	f_deriv_xprev = 1/2*A.'*(x_CG(: ,itr-1)+sigma_0*d) + 1/2*A*(x_CG(:, itr-1)+sigma_0*d) - b;
	eta_prev = f_deriv_xprev.'*d;
	
	x_CG(:, itr) = x_CG(:, itr-1);
	
	% The iteration are terminated when each update is under a tolerance
	% value or when the number of iteration is higher than jtr max
	while (jtr < jtr_max) && ((alpha_cg.^2*delta_zero) > TOL)
		f_deriv = 1/2*A.'*x_CG(:, itr) + 1/2*A*x_CG(:, itr) - b;
		eta_cg = f_deriv.'*d;
		alpha_cg = (alpha_cg*eta_cg)/(eta_prev-eta_cg);
		x_CG(:, itr) = x_CG(:, itr) + alpha_cg*d; % Secant method iteration
		eta_prev = eta_cg;
		jtr = jtr + 1;
	end
	
	f_deriv = 1/2*A.'*x_CG(:, itr) + 1/2*A*x_CG(:, itr) - b;
	r = -f_deriv;
	delta_old = delta_new;
	
	delta_mid = r.'*s;
	
	s = M\r;
	delta_new = abs(r.'*s);
	beta_cg = (delta_new - delta_mid )/delta_old; % Polak-Ribier parameter
	k = k + 1;
	if (k==size_x) || (beta_cg <= 0)
		d = s;
		k = 0;
	else
		d = s + beta_cg*d ;
	end
	itr = itr + 1;
	% Print out relative error norm
	error(itr) = abs(delta_new);
	fprintf ('Relative Error norm of iteration %d is: %5.6f \n ', itr, error(itr));
end

iteration = itr-1;
fprintf('Value of x with Conjugate Gradient Method \n');
x_CG(:, iteration)
fprintf('%f\n');

% Plot relative error norm
figure(1);
semilogy(error, 'b+:');
hold on
xlabel('Number of iterations');
ylabel('Error');
title (strcat('CG in ', num2str(size_x), ' dimensions'));
hold off

error2 = x - x_CG(:, iteration);