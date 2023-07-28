function data_set = generator_mobility_switch()
    % Add the required path
    addpath('./DC-MSS/');
    
    % Define parameters
    num_traj = 200;
    max_len = 200;
    Dt = 2;
    dt = 1;
    velocity = 0.8*sqrt(4*Dt);
    seq_len = max_len/dt;
    R = 4.04;
    
    % Generate time vector
    time = repmat((1:max_len)', 1, 1, num_traj);
    
    % Define motion generation functions
    f_confined = @(T) simMultiMotionTypeTraj(num_traj, [512,512], T-1, dt, [Dt,Dt], [R,R], [0,0], [T,T; 0,0; 0,0; 0,0], 1, 0.8);
    f_free = @(T) simMultiMotionTypeTraj(num_traj, [512,512], T-1, dt, [Dt,Dt], [0,0], [0,0], [0,0; T,T; 0,0; 0,0], 1, 0.8);
    f_direct = @(T) simMultiMotionTypeTraj(num_traj, [512,512], T-1, dt, [Dt,Dt], [0.1,0.1], [velocity,velocity], [0,0; 0,0; T,T; 0,0], 1, 0.8);
    
    % Initialize data_set
    data_set = [];
    
    for noise_level = 0:0.5:5
        for T1 = 60:20:140
            T2 = max_len - T1;
            
            % Generate and concatenate trajectories for different motion types
            [data_c_f,~] = concatenateTrajectories(f_confined(T1), f_free(T2), noise_level);
            [data_c_d,~] = concatenateTrajectories(f_confined(T1), f_direct(T2), noise_level);
            [data_f_d,~] = concatenateTrajectories(f_free(T1), f_direct(T2), noise_level);
            
            % Append to the data_set
            data_set = [data_set; data_c_f; data_c_d; data_f_d];
        end
    end
end

function [data_set, traj] = concatenateTrajectories(x1, x2, noise_level)
    % Concatenate trajectories x1 and x2 with added Gaussian noise

    % Adjust x2 to start where x1 ends
    x2 = x2 - x2(1,:,:) + x1(end,:,:);
    
    % Concatenate trajectories
    traj = [x1; x2];
    
    % Generate time vector
    time = repmat((1:size(x1,1)+size(x2,1))', 1, 1, size(x1,3));
    
    % Add Gaussian noise to the trajectory
    xy = traj + noise_level*randn(size(time));
    
    % Concatenate time and xy to form the final data
    data = cat(2, time, xy);
    
    % Convert to cell array
    data_set = squeeze(mat2cell(data, size(data,1), size(data,2), ones(size(x1,3), 1)));
end