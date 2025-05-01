function [x, t, omega_vec] = hc_sim(N_pop, N_osc, phi0, D, Adj, p_lists, x_init, sim_time, u)
% Returns trajectory x, time t, and frequencies omega_vec
%
% Inputs:
%  - N_pop: number of species per hypercycle
%  - N_osc: number of hypercycles
%  - phi0: baseline phi0
%  - D: diffusion strength
%  - Adj: adjacency matrix between oscillators
%  - p_lists: [N_pop x N_osc] matrix of p-values (one column per oscillator)
%  - x_init: initial condition vector [N_pop * N_osc x 1]
%  - sim_time: total simulation time
%  - u: complex vectors for projection

N = N_pop * N_osc;
tspan = [0 sim_time];

% Solve ODE
[t, x] = ode45(@dynamics, tspan, x_init);

% Compute observed frequencies
omega_vec = NaN(N_osc,1);
valid_idx = t > sim_time/5;
t_trimmed = t(valid_idx);
x_trimmed = x(valid_idx,:);

for k = 1:N_osc
    idx = (k-1)*N_pop + (1:N_pop);
    z = x_trimmed(:,idx) * u;
    theta = unwrap(angle(z));
    omega_vec(k) = (theta(end) - theta(1)) / (t_trimmed(end) - t_trimmed(1));
end

    function dxdt = dynamics(~, y)
        dxdt = zeros(N,1);
        for k = 1:N_osc
            idx = (k-1)*N_pop + (1:N_pop);
            xk = y(idx);
            pk = p_lists(:,k);

            % Build fitness vector f
            f = zeros(N_pop,1);
            for a = 1:N_pop
                am1 = mod(a-2,N_pop) + 1;  % cyclic a-1
                f(a) = pk(a) * xk(am1);
            end

            phi = phi0 + xk' * f;
            dx_self = xk .* (f - phi);

            % Diffusive coupling from neighbors
            dx_couple = zeros(N_pop,1);
            for j = 1:N_osc
                if Adj(k,j) > 0
                    jdx = (j-1)*N_pop + (1:N_pop);
                    dx_couple = dx_couple + Adj(k,j)*(y(jdx) - xk);
                end
            end

            dxdt(idx) = dx_self + D * dx_couple;
        end
    end

end
