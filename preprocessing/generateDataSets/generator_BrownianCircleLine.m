function data_set = generator_BrownianCircleLine(t, num_idx, mode)
% Initialize the data set
data_set = cell(num_idx, 1);

% Set default probabilities
prob = struct('line_add', 0.15, 'circle_add', 0.15, 'brownian_add', 0.15);

if nargin == 3
    % Adjust probabilities based on mode
    prob = adjust_probabilities(prob, mode);
else
    prob = adjust_probabilities(prob);
end

% Generate the data set
data_set = generate_data_set(t, num_idx, prob);
end

function prob = adjust_probabilities(prob, mode)
% Check if mode is specified
if nargin == 2
    if strcmp(mode, 'line')
        prob.line_add = 0.70;
    elseif strcmp(mode, 'circle')
        prob.circle_add = 0.70;
    else
        prob.brownian_add = 0.70;
    end
    % Calculate cumulative probabilities
    prob.brownian = prob.brownian_add;
    prob.circle = prob.brownian + prob.circle_add;
    prob.line = prob.circle + prob.line_add;
else
    % Calculate cumulative probabilities
    prob.brownian = 1/3;
    prob.circle = 2/3;
    prob.line = 1;
end


end

function data_set = generate_data_set(t, num_idx, prob)
% Initialize parameters
step = 30;
sigma = 0;
sigma_x = 0.25;
data_set = cell(num_idx, 1);

% Loop over each index
for m = 1:num_idx
    count = 0;
    v = randn(1, 2);
    while count <= t
        add_steps = floor(step * (1+rand(1)));
        count = count + add_steps;
        v = process_step(v, add_steps, sigma, prob);
    end
    data_set{m} = [(1:(count+1))', cumsum(v) + sigma_x*randn(count+1,2)];
end

end

function v = process_step(v, add_steps, sigma, prob)
% Determine the next step
px = rand(1);
if px <= prob.brownian
    v = [v; (2+sigma)*randn(add_steps, 2)];
elseif px <= prob.circle && px > prob.brownian
    v = process_circle_step(v, add_steps, sigma);
elseif px <= prob.line && px > prob.circle
    v = process_line_step(v, add_steps, sigma);
end
end

function v = process_circle_step(v, add_steps, sigma)
% Compute circular step
p = v(end, :);
pos = 2 * (det([p; 1, 0]) > 0) - 1;
theta = pos * acos(sum(p .* [1, 0]) / sqrt(sum(p .^ 2)));
dtheta = 2 * pi / 2 / 30;
theta = linspace(theta, theta + add_steps * dtheta, add_steps)';
v_norm = 1 + randn(add_steps, 2);
v_temp = v_norm .* [cos(theta), sin(theta)] + sigma * randn(add_steps, 2);
v = [v; v_temp];
end

function v = process_line_step(v, add_steps, sigma)
% Compute linear step
p = v(end, :);
pos = 2 * (det([p; 1, 0]) > 0) - 1;
theta = pos * acos(sum(p .* [1, 0]) / sqrt(sum(p .^ 2)));
v_temp = [cos(theta), sin(theta)] .* (1 + randn(add_steps, 2)) + sigma * randn(add_steps, 2);
v = [v; v_temp];
end