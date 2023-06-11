function [E_sct, Ez_inc, LeafCube_sl, LeafCube_center, H0, eps_vector] = Compute_fields(eps_0, mu_0, f, N_dis, N_scatterPoints, S1_position)
	%UNTITLED4 Summary of this function goes here
	% Detailed explanation goes here

	w = 2*pi*f;
	eta_0 = sqrt(mu_0/eps_0);
	k_0 = w*sqrt(eps_0*mu_0);
	c = 1/( sqrt(eps_0*mu_0));
	lambda_0 = c/f;
	%% Initialize characteristics of scattered problem
	% eps r1 = 2.56;
	eps_r1 = 2.0 + 1i*1.0;
	mu_r1 = 1.0;
	k_1 = w*sqrt(eps_0*eps_r1*mu_0*mu_r1);

	%% Discretize a space into 60x60 small−cubes center at (0 ,0)
	BigCube_sl = 3*lambda_0/(2*pi); % side length of a big cube
	LeafCube_sl = BigCube_sl/N_dis; % side length of a leaf cube
	LeafCube_center = zeros(N_dis*N_dis, 2); % first column is x, second is y
	LeafCube_ind = zeros(N_dis*N_dis, 1);
	LeafCube_ind(:) = [1:N_dis*N_dis];
	LeafCube_center(:,1) = mod(LeafCube_ind-1,N_dis)*LeafCube_sl+(0 - BigCube_sl/2 + LeafCube_sl/2);
	LeafCube_center(:,2) = floor((LeafCube_ind -1)/N_dis)*LeafCube_sl +(0 - BigCube_sl/2 + LeafCube_sl/2);
	clear LeafCube_ind

	%% Definition of a scatterer
	% in this example, we first consider a 2D sphere centered at(x,y) =(0 ,0)
	radius_1 = lambda_0/(2*pi);
	center_1 = [0, 0];

	%% Define scatterers
	% Exclude cubes outside the cylinder
	eps_vector = zeros(N_dis*N_dis, 1);
	Rel_position = LeafCube_center - repmat(center_1, [N_dis*N_dis 1]);
	Rel_position_norm = sqrt(Rel_position(:, 1).^2 + Rel_position(:, 2).^2);
	eps_vector(find(Rel_position_norm < radius_1)) = eps_r1;
	clear Rel_position

	%% Initialize incident wave
	% in this code, we only investigate a TM wave which has only Ez component
	Ez_inc = zeros(N_dis*N_dis, 1);
	% if the incident wave is produced by a line source located at S1 position
	Rel_position = LeafCube_center - repmat(S1_position, [N_dis*N_dis 1]);
	Rel_position_norm = sqrt(Rel_position(:, 1).^2 + Rel_position(:, 2).^2);
	Ez_inc = 1*exp(-1i*k_0*Rel_position_norm);

	%% Generate an impedance matrix
	equ_radius = sqrt(LeafCube_sl*LeafCube_sl/pi);

	Reduced_LeafCubeIndex = find(eps_vector~=0);
	Reduced_eps_vector = eps_vector(Reduced_LeafCubeIndex);
	Reduced_Ezinc = Ez_inc(Reduced_LeafCubeIndex);
	Reduced_LeafCube_center = LeafCube_center(Reduced_LeafCubeIndex, :);
	clear Reduced_LeafCubeIndex

	SizeOfSystem = length(Reduced_eps_vector);

	Z_impedance = zeros(SizeOfSystem, SizeOfSystem);

	% off-diagonal elements
	for cnt1=1:SizeOfSystem
		Rel_position = repmat(Reduced_LeafCube_center(cnt1, :), [SizeOfSystem, 1]) - Reduced_LeafCube_center;
		Rel_position_norm = sqrt(Rel_position(:, 1).^2 + Rel_position(:, 2).^2);
		index = find(Rel_position_norm~=0);
		Z_impedance(cnt1, index) = eta_0*pi*equ_radius/2.0*besselj(1, k_0*equ_radius)*besselh(0 ,2 ,k_0*Rel_position_norm(index));
	end
	clear Rel_position Rel_position_norm index

	% diagonal elements
	DiagZ = zeros(SizeOfSystem, 1);
	DiagZ(:) = eta_0*pi*equ_radius/2.0*besselh(1, 2, k_0*equ_radius) - 1i*eta_0*Reduced_eps_vector./(k_0*(Reduced_eps_vector -1));
	for cnt1 =1:SizeOfSystem
		Z_impedance(cnt1, cnt1) = DiagZ(cnt1);
	end
	clear DiagZ

	%% Compute the current
	J = Z_impedance\Reduced_Ezinc;

	%% Compute the scattered field
	radius_scat = 2*lambda_0;
	for cnt1 = 1:N_scatterPoints
		scatterPosition(:, cnt1) = [radius_scat*pi*cos(2*pi/N_scatterPoints*cnt1); radius_scat*pi*sin(2*pi/N_scatterPoints*cnt1)];
	end

	for cnt1=1:N_scatterPoints
		vector = Reduced_LeafCube_center - repmat(scatterPosition(:, cnt1).',[size(Reduced_LeafCube_center, 1) 1]);
		scatObj_dis(:, cnt1) = sqrt(vector(:, 1).^2 + vector(:, 2).^2); % Distance between each Leaf Cube center and each scatter points
	end

	% Scattered Field at every scatter points
	E_sct = sum((-1/4*w*mu_0)*repmat(J, [1 N_scatterPoints]).*besselh(0, 2, k_0*scatObj_dis)*LeafCube_sl*LeafCube_sl, 1);
	E_sct = E_sct.';

	% Value of the hankel function when the position between two points is zero
		a = equ_radius;
		H1_1 = 2*pi*a*besselh(1 ,2 ,k_0*a)/k_0;
		H1_2 =(4*1i*Reduced_eps_vector(1, 1))/(k_0.^2*(Reduced_eps_vector(1, 1)-1));
		H1_3 = 4/(1i*k_0.^2*(Reduced_eps_vector(1, 1)-1));
		H0 = H1_1 - H1_2 - H1_3;
end