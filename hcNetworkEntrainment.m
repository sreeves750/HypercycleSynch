function hcNetworkEntrainment

N_pop = 10;                  % 10 populations per hypercycle
N_osc = 20;                  
N = N_pop * N_osc;           % total populations
phi0 = 0.0005;
p_edge = 3/N_osc;                  % ER link probability
sim_time = 1000;
D_range = logspace(-12, -1.5, 30);  % diffusion strengths to sweep

% Keep generating ER networks until the entire graph is connected
connected = false;
while ~connected
    node = makeER(N_osc, p_edge);
    Adj = adjacency(node);

    % Check if the graph is fully connected
    G = graph(Adj > 0);
    bins = conncomp(G);
    num_components = max(bins);

    if num_components == 1
        connected = true;
    end
end

Adj = Adj ./ max(1, sum(Adj,2));  % normalize rows

% Interaction parameters for each hypercycle
p_mean = 57.415;
p_lists = p_mean * (1 + 0.1*randn(N_pop, N_osc));

% Initial conditions
x_init = rand(N,1);
for k = 1:N_osc
    idx = (k-1)*N_pop + (1:N_pop);
    x_init(idx) = x_init(idx) / sum(x_init(idx));
end

u = exp(2*pi*1i*(0:N_pop-1)'/N_pop);

omega_matrix = NaN(length(D_range), N_osc);

% Parallel progress tracker
q = parallel.pool.DataQueue;
progress = 0;
Nsteps = length(D_range);
afterEach(q, @trackProgress);

parfor k = 1:Nsteps
    D = D_range(k);
    [~, ~, omega_vec] = hc_sim(N_pop, N_osc, phi0, D, Adj, p_lists, x_init, sim_time, u);
    omega_matrix(k,:) = omega_vec;
    send(q, 1);
end

% Plot observed frequencies
figure(1); clf; hold on;
ch = lines(N_osc);
for i = 1:N_osc
    plot(D_range, omega_matrix(:,i), '-', 'Color', ch(i,:), 'LineWidth', 1.5);
end
set(gca, 'XScale', 'log');
xlabel('Diffusion strength D');
ylabel('Observed frequency \omega_i');
grid on;
set(gcf,'Color','w');

    function trackProgress(~)
        progress = progress + 1;
        if mod(progress, 2) == 0 || progress == Nsteps
            fprintf('Progress: %d of %d (%.1f%%)\n', progress, Nsteps, 100*progress/Nsteps);

            % Plot the current results so far
            figure(10); clf; hold on;
            ch = lines(N_osc);
            for i = 1:N_osc
                plot(D_range(1:progress), omega_matrix(1:progress,i), '-', 'Color', ch(i,:), 'LineWidth', 1.5);
            end
            set(gca, 'XScale', 'log');
            xlabel('Diffusion strength D');
            ylabel('Observed frequency \omega_i');
            title(sprintf('Progress after %d steps', progress));
            grid on;
            set(gcf, 'Color', 'w');
            drawnow;
        end
    end

end
