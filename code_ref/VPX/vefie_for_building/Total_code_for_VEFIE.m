%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                                                                   %%%
%%%     Solving Scattering problem for a building shape scatterer     %%%
%%%                Using 2D FFT and reduced CG algorigm               %%%
%%%                                                                   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

data % gives some used variables and fix size of region of interest

create_building_shape % specify space grid with materials

specify_interior %posisiion, k and rho, for each basis counter

make_VEFIE_elements % creates G_vector, D and V from (I+GD)E=V

solve_with_FFT_and_ReducedCG

plot_results

