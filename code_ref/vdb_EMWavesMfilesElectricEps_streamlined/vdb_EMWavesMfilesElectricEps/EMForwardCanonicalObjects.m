clear all; clc; close all; clear workspace
input = initEM();

% Compute scattered acoustic field at receivers around circular cylinder
disp('Running EMsctCircle');
EMsctCircle;
