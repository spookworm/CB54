%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                                                                     %%%
%%%  CREATES BUILDING SHAPE                                             %%%
%%%  1) Discretisize the field of interest (Spacial N X M grids)        %%%
%%%  2) Material specification for spacial grid points                  %%%    
%%%                                                                     %%%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 1) discretisize field of interest, imag = Yaxes, real = Xaxes %%%%%%%%%

disc_per_lambda = 10; % chosen accuracy
length_x_side = 30; % in meters 
length_y_side = 20; % in meters

centre = 0.0 + 0.0*j ; 
start_pt = centre - 0.5*length_x_side - 0.5*length_y_side*j ; 
 
N = floor(length_x_side/(abs(lambda_d)/disc_per_lambda)); % force N = multp 4
fourth_of_N =  ceil(N/4);   
while(mod(N,fourth_of_N) ~= 0 )
    N = N + 1; 
end
delta_x = length_x_side/N;

M = floor(length_y_side/(delta_x)); % force M = multp 4, size dy near dx
fourth_of_M =  ceil(M/4);   
while(mod(M,fourth_of_M) ~= 0 )
    M = M + 1; 
end
delta_y = length_y_side/M ; 

equiv_a = sqrt(delta_x*delta_y/pi);

fprintf('\n \nSize of field of interst is %d meter by %d meter \n',length_x_side,length_y_side)
fprintf('Discretizised space is %d grids by %d grids\n',N,M)
fprintf('with grid size %d meter by %d meter \n',delta_x, delta_y)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% 2) Material specification for the building %%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%% specify some sizes so we can scale up

hN = floor(N/2); % half of N
tN = floor(N/3); % thirt of N
qN = floor(N/4); % quarter of N
fN = floor(N/5); % fifth of N
tnN = floor(N/10);

hM = floor(M/2); 
tM = floor(M/3); 
qM = floor(M/4); 
fM = floor(M/5);


dX = round(0.75/delta_x); %door size x direction
dY = round(0.75/delta_y); %door size y direction
wX = round(0.2/delta_x); %wall size x direction
wY = round(0.2/delta_y); %wall size y direction


%%%%%%%%%%%%%%%%%%%% Specify building (X1:X2,Y1:Y2)
% 1) concrete, 2) wood, 3) glass

building = zeros(N,M);

building(1:N,1:1+wY)=1;
building(1:N,M:-1:M-wY)=1;
building(1:1+wX,1:M)= 1;
building(N:-1:N-wX,1:M)=1;

building(2*tN-wX:2*tN,1:M)=1; % rechts
building(2*tN:2*tN+tnN,2*tM:2*tM+wY)=1; 
building(N-wX-tnN:N-wX,2*tM:2*tM+wY)=1;
building(2*tN-wX:2*tN,2*tM+wY+dY:2*tM+wY+2*dY)=0; %deur gat boven
building(2*tN-wX+1:2*tN,2*tM+wY+dY:2*tM+wY+2*dY)=2; %deur boven
building(2*tN-wX:2*tN,2*tM-wY-2*dY:2*tM-wY-dY)=0; %deur gat beneden
building(2*tN-wX+1:2*tN,2*tM-wY-2*dY:2*tM-wY-dY)=2; %deur beneden

building(hN-wX:hN,2*tM:M)=1; %links boven
building(1:hN,2*tM:2*tM+wY)=1;
building(fN:fN+dX,2*tM:2*tM+wY)=0;
building(fN:fN+dX,2*tM+1:2*tM+wY)=2;

building(hN+2*wX:2*tN-3*wX,M-wY:M)=0; %grote deur
building(hN+2*wX:2*tN-3*wX,M-wY+1:M)=2;

building(1:2*tN-wX,tM:tM+wY)=1; %onder links
building(tN:tN+wX,1:tM)=1;
building(fN:fN+dX,tM:tM+wY)=0; %deur links
building(fN:fN+dX,tM:tM+wY-1)=2; 
building(hN+1-dX:hN+1,tM:tM+wY)=0; %deur rechts
building(hN+1-dX:hN+1,tM:tM+wY-1)=2; 

building(1+wX+dX:tN-dX,1:1+wY)=3;
building(tN+wX+dX:2*tN-dX,1:1+wY)=3;
building(2*tN+wX+dX:N-dX-2*wX,1:1+wY)=3;
building(N-wX:N,2*wY+dY:2*tM-2*wY-dY)=3;
building(N-wX:N,2*tM+wY+dY:M-wY-dY)=3;
building(2*tN+wX+dX:N-dX-2*wX,M-wY:M)=3;
building(1+wX+dX:hN-wX-dX,M-wY:M)=3;

building = building.'; %translate to (row = y,collom = x)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% show building
if 1==1
figure
    surf([1:N]*delta_x,[1:M]*delta_y,building)
view(2)
shading interp
xlabel('Distance in meters')
ylabel('Distance in meters')
end
