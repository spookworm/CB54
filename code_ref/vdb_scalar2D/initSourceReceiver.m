function input = initSourceReceiver(input)
input.xS(1) = -170; % source position
input.xS(2) = 0;

input.NR = 180; % receiver positions
input.rcvr_phi(1:input.NR) = (1:input.NR) * 2 * pi / input.NR;
input.xR(1, 1:input.NR) = 150 * cos(input.rcvr_phi);
input.xR(2, 1:input.NR) = 150 * sin(input.rcvr_phi);