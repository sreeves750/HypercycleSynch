function hcFreq

N = 5;
phi0 = 0.0005;
sim_time = 1000;
p = 9.355; 
% N=5 -> p=9.355
% N=6 -> p=15.57
% N=7 -> p=23.29
% N=8 -> p=32.71
% N=9 -> p=43.88
% N=10 -> p=57.41
% N=11 -> p=73.56
% N=12 -> p=91.37
% N=13 -> p=115.45
% N=14 -> p=139.55
% N=15 -> p=174.46
% N=16 -> p=206.14
% N=17 -> p=258.31
% N=18 -> p=293.3
% N=19 -> p=371.39
% N=20 -> p=427.62


u = exp(2*pi*1i*(0:N-1)'/N);

% Define p_list for hypercycle
p_list = p * ones(N,1);

% Initial condition
x1i = rand(N,1); 
x1i = x1i / sum(x1i);

omega1_free = hc_uncoupled_freq(N, phi0, p_list, x1i, sim_time, u);
fprintf('omega1 = %0.10f\n', omega1_free);

end