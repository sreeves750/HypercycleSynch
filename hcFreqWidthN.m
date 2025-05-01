function hcFreqWidthN

N_values = 5:20;
% Table for p values that give frequency of 1
p_table = [9.355, 15.57, 23.29, 32.71, 43.88, 57.41, 73.56, 91.37, 115.45, 139.55, 174.46, 206.14, 258.31, 293.3, 371.39, 427.62];

phi0 = 0.0005;
sim_time = 1000;
searches = 10;
threshold = 0.001;

a_array = NaN(size(N_values));
b_array = NaN(size(N_values));

figure(100); clf;

for n_idx = 1:length(N_values)
    N_pop = N_values(n_idx);
    p = p_table(n_idx);

    num_repeats = 3;
    a_repeats = NaN(num_repeats,1);
    b_repeats = NaN(num_repeats,1);

    for rep_idx = 1:num_repeats
        D_values = logspace(-3*N_pop, -1, 24);
        widths = NaN(size(D_values));

        u = exp(2*pi*1i*(0:N_pop-1)'/N_pop);
        Adj = [0 1; 1 0];
        p_list_base = p * ones(N_pop,1);

        x1i = rand(N_pop,1); x1i = x1i / sum(x1i);
        x2i = rand(N_pop,1); x2i = x2i / sum(x2i);
        x_init = [x1i; x2i];

        q = parallel.pool.DataQueue;
        Nsteps = length(D_values);
        afterEach(q, @(~) trackProgress(Nsteps));

        parfor d_idx = 1:Nsteps
            diffusion_strength = D_values(d_idx);

            p_list_base_local = p_list_base;
            x_init_local = x_init;
            u_local = u;
            Adj_local = Adj;
            x1i_local = x1i;
            x2i_local = x2i;

            right_offset = NaN;
            offset_guess = 0;
            while offset_guess <= 10
                [locked, ~] = check_locking(offset_guess, diffusion_strength, p_list_base_local, x_init_local, N_pop, 2, phi0, sim_time, Adj_local, u_local, threshold);
                if locked
                    offset_guess = offset_guess + 0.5;
                else
                    break;
                end
            end
            if offset_guess > 10
                right_offset = NaN;
            else
                low = offset_guess - 0.5;
                high = offset_guess;
                for iter = 1:searches
                    mid = (low + high)/2;
                    [locked, ~] = check_locking(mid, diffusion_strength, p_list_base_local, x_init_local, N_pop, 2, phi0, sim_time, Adj_local, u_local, threshold);
                    if locked
                        low = mid;
                    else
                        high = mid;
                    end
                end
                right_offset = (low + high)/2;
            end

            % Find left edge
            low = -1;
            high = 0;
            for iter = 1:searches
                mid = (low + high)/2;
                [locked, ~] = check_locking(mid, diffusion_strength, p_list_base_local, x_init_local, N_pop, 2, phi0, sim_time, Adj_local, u_local, threshold);
                if locked
                    high = mid;
                else
                    low = mid;
                end
            end
            left_offset = (low + high)/2;

            if ~isnan(left_offset) && ~isnan(right_offset)
                omega_left = hc_uncoupled_freq(N_pop, phi0, p_list_base_local * (1 + left_offset), x2i_local, sim_time, u_local);
                omega_right = hc_uncoupled_freq(N_pop, phi0, p_list_base_local * (1 + right_offset), x2i_local, sim_time, u_local);
                widths(d_idx) = omega_right - omega_left;

                if (widths(d_idx) < 0)
                    widths(d_idx) = NaN;
                end
            else
                widths(d_idx) = NaN;
            end

            send(q, 1);
        end


        xdata = D_values(:);
        ydata = widths(:);
        valid = ~isnan(xdata) & ~isnan(ydata);
        xdata = xdata(valid);
        ydata = ydata(valid);

        fitmodel = fittype('a*log(1 + x*exp(b))', 'independent', 'x', 'coefficients', {'a', 'b'});
        opts = fitoptions('Method', 'NonlinearLeastSquares', 'StartPoint', [1/10, 10]);

        [fitresult, ~] = fit(xdata, ydata, fitmodel, opts);

        a_repeats(rep_idx) = fitresult.a;
        b_repeats(rep_idx) = fitresult.b;
    end

    % Average over the trials
    a_array(n_idx) = mean(a_repeats);
    b_array(n_idx) = mean(b_repeats);

     % Live plot
    figure(100);
    subplot(2,1,1);
    plot(N_values(1:n_idx), a_array(1:n_idx), 'ko-','MarkerFaceColor','k'); hold on;
    xlabel('N'); ylabel('a'); grid on;

    subplot(2,1,2);
    plot(N_values(1:n_idx), b_array(1:n_idx), 'bo-','MarkerFaceColor','b'); hold on;
    xlabel('N'); ylabel('b'); grid on;

    drawnow;
end


fprintf('Done sweeping N!\n');

% Fit a(N) ~ 1/(cN+d)
fit_a = fit(N_values(:), a_array(:), '1./(c*x + d)', 'StartPoint', [1, 3]);

% Fit b(N) ~ e*N + f
fit_b = fit(N_values(:), b_array(:), 'poly1');

figure(100);

subplot(2,1,1);
plot(N_values, fit_a(N_values), 'r-', 'LineWidth', 1.5);
legend('Data','Fit: 1/(cN+d)');

subplot(2,1,2);
plot(N_values, fit_b(N_values), 'r-', 'LineWidth', 1.5);
legend('Data','Fit: eN+f');

fprintf('Fitted a(N): c = %.6f, d = %.6f\n', fit_a.c, fit_a.d);
fprintf('Fitted b(N): e = %.6f, f = %.6f\n', fit_b.p1, fit_b.p2);


end

function [locked, omega_diff] = check_locking(offset, diffusion_strength, p_list_base, x_init, N_pop, N_osc, phi0, sim_time, Adj, u, threshold)
    p_lists = [p_list_base, p_list_base * (1 + offset)];
    [~, ~, omega_vec] = hc_sim(N_pop, N_osc, phi0, diffusion_strength, Adj, p_lists, x_init, sim_time, u);
    if any(isnan(omega_vec))
        locked = false;
        omega_diff = NaN;
    else
        omega_diff = omega_vec(1) - omega_vec(2);
        locked = abs(omega_diff) < threshold;
    end
end

function trackProgress(Nsteps)
    persistent progress_internal
    if isempty(progress_internal)
        progress_internal = 0;
    end
    progress_internal = progress_internal + 1;
    if mod(progress_internal, 6) == 0 || progress_internal == Nsteps
        fprintf('Progress: %d of %d (%.1f%%)\n', progress_internal, Nsteps, 100*progress_internal/Nsteps);
    end
end