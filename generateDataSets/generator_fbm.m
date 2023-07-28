function [data_set, information] = generator_fbm()
    % Define a set of Hurst values
    % The wfbm function in MATLAB generates standard fractional Brownian motion (fBm). 
    
    Hurst_set = [0.5, 1.0, 1.5];
    
    % Define the length of the fractional Brownian motions
    len = 2e4;
    
    % Initialize the data_set cell array
    data_set = {};
    
    % Convert degrees to radians for later calculations
    thetaTopi = pi /180; % rad
    
    for Hurst = Hurst_set
        % Generate fractional Brownian motions for x and y
        x = wfbm(Hurst/2, len)'; 
        vx = diff(x);
        y = wfbm(Hurst/2, len)'; 
        vy = diff(y);
        
        % Normalize the x and y coordinates
        XY = [x/std(vx), y/std(vy)];
        
        % Generate a time vector
        t = (1:size(XY, 1))';
        
        % Append the time, x, and y data to the data_set cell array
        data_set(end+1, 1) = {[t, XY]};
    end
    
    % Generate and normalize three sets of fractional Brownian motions for theta
    theta_f = normalize_v(wfbm(0.25, len)', 3*thetaTopi);
    theta_b = normalize_v(wfbm(0.5, len)', 3*thetaTopi);
    theta_l = normalize_v(wfbm(0.005, len)', 3*thetaTopi);
    
    % Compute the information matrix
    information = {abs(bound_range(theta_f, 10*thetaTopi))/thetaTopi;...
        abs(bound_range(theta_b, 90*thetaTopi))/thetaTopi;...
        abs(bound_range(theta_l, 10*thetaTopi))/thetaTopi};
end

function y = normalize_v(x, scale)
    % Normalize a vector x by its standard deviation
    % If a scale factor is provided, scale the vector by this factor
    
    if nargin < 2
        scale = 1;
    end
    
    v = diff(x);
    v_std = sqrt(sum(var(v), 2));
    
    y = cumsum(scale*[0; v]/v_std);
    y = y - mean(y);
end

function y = bound_range(x, bound)
    % Bound a range of angles between -pi and pi, and then between -bound and bound
    
    % Project to [-pi, pi]
    x = sign(x).*mod(abs(x), pi); 
    
    x_val = mod(abs(x), bound); 
    x_orient = mod(floor(abs(x)/bound), 2);

    y = sign(x).*(x_orient*bound + ((1-2*x_orient)).*x_val);
end