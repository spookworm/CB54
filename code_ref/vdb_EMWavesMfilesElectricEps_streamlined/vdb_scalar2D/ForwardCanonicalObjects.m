clear all; clc; close all; clear workspace
input = init();

% Compute scattered wave at receivers around the circular cylinder
disp('Running WavefieldSctCircle');
WavefieldSctCircle;
