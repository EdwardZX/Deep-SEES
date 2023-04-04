classdef GetSingleParamsBase < handle
    %GETSINGLEPARAMSBASE Summary of this class goes here
    %   Detailed explanation goes here
    methods(Abstract)
        get_params(obj)
        draw(obj)
        get_distribution_spatial(obj)
        draw_spatial(obj)
    end
end

