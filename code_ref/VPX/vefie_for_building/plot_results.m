%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                                                                     %%%
%%%    Write variable information to Command Window and plot reults     %%%
%%%                                                                     %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WRITE INFORMATION TO COMMAND WINDOW %%%

fprintf('\nUsed frequency of incomming wave = %d MHz \n', f/1000000)

fprintf('\nRelative Epsilon: \nconcrete = %d \nwood = %d \nglass = %d \n',epsilonr,epsilonrw,epsilonrg)
fprintf('\nWave Lenghts: \nfree space %d meter\nconcrete %d meter\nwood %d meter \nglass %d meter\n',lambda0,lambda_d,lambda_w,lambda_g)

fprintf('\n \nSize of field of interst is %d meter by %d meter \n',length_x_side,length_y_side)
fprintf('Discretizised space is %d grids by %d grids\n',N,M)
fprintf('with grid size %d meter by %d meter',delta_x, delta_y)
fprintf('So basis counter goes to %d \n', basis_counter)

fprintf('\nNumber of unknowns in Z is than %d\n',N*M*N*M)
fprintf('Number of reduced unknowns is %d\n',sum(Rfo)*N*M)
fprintf('with %d percent of the is filled by contrast \n',sum(Rfo)/(N*M)*100)

fprintf('\nCG iteration error tollerance = %d \n',tol)
fprintf('Duration of reduced CG iteration = %d seconds \n',time_Red)

%%%%%%%%%%%%%%%%%%%%%%%% CONVERT SOLUTIONS ON 2D GRID, FOR 3D PLOTS %%%

for i = 1:M %without vec2mat:
    vec2matSimulationEred(i,:) = Ered((i-1)*N+1:i*N);
    vec2matSimulationVred(i,:) = Vred((i-1)*N+1:i*N);
end
Ered_2D = vec2matSimulationEred;
Vred_2D = vec2matSimulationVred;
ScatRed_2D = Ered_2D - Vred_2D;
x = real(position(1:N));
y = imag(position(1:N:N*M));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CREATE ALL THE POTS %%

if 1 == 1 % Error en configuration
figure

subplot(1,2,1)
    plot(Concrete_configuration,'r*')
    hold on
    plot(Wood_configuration,'b*')
    hold on
    plot(Glass_configuration,'y*')
axis([-0.5*length_x_side 0.5*length_x_side -0.5*length_y_side 0.5*length_y_side])
title('Building configuration')
legend('Concrete','Wood eps','glass')
xlabel('x (meters)')
ylabel('y (meter)')

subplot(1,2,2)
    semilogy(Reduced_iteration_error, 'r')
title('Iteration Error')
legend('Reduced')
ylabel('Number of iteration')
xlabel('Error')
end

if 1 == 1 % 3D plots of real/abs of total/scatterd/incomming
figure 

subplot(2,3,1)
    surf(x,y,real(Vred_2D));
view(2)
shading interp
title('Real part reduced incomming wave');
xlabel('x (meters)')
ylabel('y (meter)')

subplot(2,3,2)
    surf(x,y,real(ScatRed_2D));
view(2)
shading interp
title('Real part reduced scattered field');
xlabel('x (meters)')
ylabel('y (meter)')

subplot(2,3,3)
    surf(x,y,real(Ered_2D)) 
view(2) 
shading interp
title('Real part reduced total field')
xlabel('x (meters)')
ylabel('y (meter)')

subplot(2,3,4) 
    surf(x,y,abs(Vred_2D));
view(2)
shading interp
title('Absoluut part reduced incomming wave');
xlabel('x (meters)')
ylabel('y (meter)')

subplot(2,3,5)
    surf(x,y,abs(ScatRed_2D));
view(2)
shading interp
title('Absoluut part reduced scattered field');
xlabel('x (meters)')
ylabel('y (meter)')

subplot(2,3,6)
    surf(x,y,abs(Ered_2D)) 
view(2) 
shading interp
title('Absuluut part reduced total field')
xlabel('x (meters)')
ylabel('y (meter)')

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% END

