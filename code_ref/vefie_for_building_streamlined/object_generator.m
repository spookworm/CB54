function object = object_generator(materials,M,N,delta_x,delta_y,input)
% CREATE_object_SHAPE: BUILD THE ORIGINAL object LAYOUT PROGRAMMATICALLY

%%% 2) Material specification for the object %%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%% Specify object (X1:X2,Y1:Y2)
% 1) vacuum; 2) concrete; 3) wood; 4) glass;
object = ones(N, M);

object(1:N, 1:1+wY) = 2;
object(1:N, M:-1:M-wY) = 2;
object(1:1+wX, 1:M) = 2;
object(N:-1:N-wX, 1:M) = 2;

object(2*tN-wX:2*tN, 1:M) = 2; % rechts
object(2*tN:2*tN+tnN, 2*tM:2*tM+wY) = 2;
object(N-wX-tnN:N-wX, 2*tM:2*tM+wY) = 2;
object(2*tN-wX:2*tN, 2*tM+wY+dY:2*tM+wY+2*dY) = 1; %deur gat boven
object(2*tN-wX+1:2*tN, 2*tM+wY+dY:2*tM+wY+2*dY) = 3; %deur boven
object(2*tN-wX:2*tN, 2*tM-wY-2*dY:2*tM-wY-dY) = 1; %deur gat beneden
object(2*tN-wX+1:2*tN, 2*tM-wY-2*dY:2*tM-wY-dY) = 3; %deur beneden

object(hN-wX:hN, 2*tM:M) = 2; %links boven
object(1:hN, 2*tM:2*tM+wY) = 2;
object(fN:fN+dX, 2*tM:2*tM+wY) = 1;
object(fN:fN+dX, 2*tM+1:2*tM+wY) = 3;

object(hN+2*wX:2*tN-3*wX, M-wY:M) = 1; %grote deur
object(hN+2*wX:2*tN-3*wX, M-wY+1:M) = 3;

object(1:2*tN-wX, tM:tM+wY) = 2; %onder links
object(tN:tN+wX, 1:tM) = 2;
object(fN:fN+dX, tM:tM+wY) = 1; %deur links
object(fN:fN+dX, tM:tM+wY-1) = 3;
object(hN+1-dX:hN+1, tM:tM+wY) = 1; %deur rechts
object(hN+1-dX:hN+1, tM:tM+wY-1) = 3;

object(1+wX+dX:tN-dX, 1:1+wY) = 4;
object(tN+wX+dX:2*tN-dX, 1:1+wY) = 4;
object(2*tN+wX+dX:N-dX-2*wX, 1:1+wY) = 4;
object(N-wX:N, 2*wY+dY:2*tM-2*wY-dY) = 4;
object(N-wX:N, 2*tM+wY+dY:M-wY-dY) = 4;
object(2*tN+wX+dX:N-dX-2*wX, M-wY:M) = 4;
object(1+wX+dX:hN-wX-dX, M-wY:M) = 4;

object = object.'; %translate to (row = y,collom = x)
% writematrix(object, './Geometry/object.txt')
writematrix(object, [input.directory_geom input.object_name])
% imwrite(uint8(object), [input.directory_geom input.object_name])
end