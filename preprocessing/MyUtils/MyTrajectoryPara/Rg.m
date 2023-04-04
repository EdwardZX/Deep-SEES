classdef Rg < GetSingleParamsBase
    %MSD Summary of this class goes here
    %   Detailed explanation goes here
    methods
        function results = get_params(obj,tjObj)
            %Rg Construct an instance of this class
            slices = tjObj.get_slices();            
            results.Rg =  arrayfun(@(vec) obj.cal_rg(vec),slices);
            %plot(results)            
        end         
        
        function result = cal_rg(obj,vec)
            %get the single slices msd
            % the len of single slice must be twice longer than lag
            vec = vec{1};
            result = mean(sum((vec - mean(vec)).^2,2));          
        end
        
        function  draw(obj,paramObj)
            all_data_set = [];
            %expression = ['paramObj.trajectory_params{m}','.',val];
            for m = 1:numel(paramObj.trajectory_params)
                all_data_set = [all_data_set;paramObj.trajectory_params{m}.Rg]; 
            end
%            index = find(all_data_set_alpha < 1);
%            all_data_set_alpha(index) = [];
%            all_data_set_Dt(index) = [];
           %paramObj.plot_log_pdf(all_data_set_alpha);
           paramObj.plot_log_pdf(all_data_set);
           hold on
           xlabel('{R_g}'); ylabel('{log PDF}')
          % legend(paramObj.params.legends)
        end
        
        function get_distribution_spatial(obj) 
        end
        
        function draw_spatial(obj)
        end
        
    end
end





