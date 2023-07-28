function data_set = fast_slow_variant()
    P = eye(4); % Identity matrix
    dt = 1/30; % Time step
    D = [0.006,0.025,0.1,0.4]; % Diffusion constants
    num = 20000; % Number of iterations
    p_slow = [0.5;0.45;0.05;0]; % Slow process probabilities
    p_fast = [0;0;0.5;0.5]; % Fast process probabilities
    n_scale = 1; % Noise scale factor

    % Creating Markov Chains
    mc_slow = create_markov_chain(P, p_slow);
    mc_fast = create_markov_chain(P, p_fast);
    
    % Simulating the trajectories
    [X_slow, y_slow] = simulate_trajectory(mc_slow, num, D, dt, [1,0,0,0]);
    [X_fast, y_fast] = simulate_trajectory(mc_fast, num, D, dt, [0,0,0,1]);
    
    % Aligning the fast trajectory with the end of the slow trajectory
    y_fast = y_fast-y_fast(1,:)+y_slow(end,:);
    
    % Creating the raw data set
    XY_raw = [y_slow;y_fast];
    
    % Adding noise to the data
    sigma = n_scale * sqrt(2 * mean(D) * dt);
    XY = XY_raw + sigma * randn(size(XY_raw));
    
    % Time vector
    t = (1:size(XY,1))';
    
    % Saving data
    data_set = {[t,XY]};
end

function mc = create_markov_chain(P, p)
    P_new = 0.9 * P + 0.1 * p;
    P_new = P_new';
    mc = dtmc(P_new);
end

function [X, y] = simulate_trajectory(mc, num, Ds, dt, X0)
    X = simulate(mc, num-1, 'X0', X0);
    states = unique(X);
    v = zeros(size(X,1),2);
    for m = 1:size(states,1)
        randn('seed',m);
        D = Ds(states(m));
        sigma = sqrt(2*D*dt);
        idx = (X == states(m));
        v(idx,:) = sigma*randn(sum(idx),2);
    end
    y = cumsum(v,1);
end