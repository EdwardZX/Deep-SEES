function all_data_set = any_parameters(paramObj,var,varargin)
all_data_set = [];

if ischar(var)
    for m = 1:numel(paramObj.trajectory_params)
        all_data_set = [all_data_set;paramObj.trajectory_params{m}.(var)];
    end
else
    for m = 1:numel(var)
        all_data_set = [all_data_set;var{m}];
    end
end

%expression = ['paramObj.trajectory_params{m}','.',val];
if nargin ==2
  paramObj.plot_log_pdf(all_data_set);
hold on      
end

end