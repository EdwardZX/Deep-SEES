classdef Autoc < GetSingleParamsBase
    %AUTOC Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        flag = 0 % whether vector false;true
        basic_instance
    end
    
    methods
        
        function obj = Autoc(basic_instance,varargin)
            if nargin == 2
                obj.flag= varargin{1};
            end
            obj.basic_instance = basic_instance;
            
        end
        
        
        function results = get_params(obj,tjObj)
            % temporal autoc function
            if obj.flag
                %result =  basic_instance.get_params(tjObj);
                v = [tjObj.xy(2:end,:)-tjObj.xy(1:end-1,:)]/tjObj.dt;
                %v = v(:);
            else
                v = obj.basic_instance.get_params(tjObj).v;
            end
            omega = obj.basic_instance.get_params(tjObj).omega;
            turn_angle = obj.basic_instance.get_params(tjObj).turn_angle;
%             M_v = size(v,1);
%             temp_v = zeros(2*M_v-1,1);
%             for cols = 1:size(v,2)
%                 %temp_v = xcorr(v(:,cols));
%                 temp_v =  temp_v + xcorr(v(:,cols));
%             end
            %M_v = size(v,1);
            %temp_v = xcorr(v);
            results.autoc_v = obj.autoc_vector(v);
            results.autoc_omega = obj.autoc_vector(omega);
            results.autoc_turn_angle = obj.autoc_vector(turn_angle);
            %plot(results.autoc_v)
%             M_omega = size(omega,1);
%             temp_omega = xcorr(omega);
%             results.autoc_omega = temp_omega(M_omega:end) / temp_omega(M_omega);
            %plot(results.autoc_omega)
        end
        
        function draw(obj,paramObj)
            %% draw the len and points of all tracks
            
            len_max = 120; ld = 1.5;
            %expression = ['paramObj.trajectory_params{m}','.',val];
            len = max(arrayfun(@(x) length(paramObj.trajectory_params{x}.autoc_v), ...
                1:numel(paramObj.trajectory_params)));
            len = min(len_max,len);
            all_data_set_v = cell2mat(arrayfun(@(x) obj.padding(paramObj.trajectory_params{x}.autoc_v,len),...
                1:numel(paramObj.trajectory_params),'UniformOutput',false));
            all_data_set_omega = cell2mat(arrayfun(@(x) obj.padding(paramObj.trajectory_params{x}.autoc_omega,len),...
                1:numel(paramObj.trajectory_params),'UniformOutput',false));
            
            all_data_set_turn_angle = cell2mat(arrayfun(@(x) obj.padding(paramObj.trajectory_params{x}.autoc_turn_angle,len),...
                1:numel(paramObj.trajectory_params),'UniformOutput',false));
            
            %            index = find(all_data_set_alpha < 1);
            %            all_data_set_alpha(index) = [];
            %            all_data_set_Dt(index) = [];
            t = (1:len)'*paramObj.params.dt*paramObj.params.window;
            subplot(3,1,1)
            %paramObj.plot_log_pdf(all_data_set_alpha);
            %plot_v =nanmean(all_data_set_v,2);
            %v_errorbar = nanstd(all_data_set_v,2);
            %             errorbar((1:len)'*paramObj.params.dt,nanmean(all_data_set_v,2),...
            %                 nanstd(all_data_set_v,0,2)*0.01,'LineWidth',1.5);
            plot((1:len)'*paramObj.params.dt,nanmean(all_data_set_v,2)...
                ,'LineWidth',ld)
            hold on
            ylabel('{ v -autoc}');xlabel('{t}')
            subplot(3,1,2)
            
            plot((1:len)'*paramObj.params.dt,nanmean(all_data_set_turn_angle,2)...
                ,'LineWidth',ld)
            hold on
            ylabel('{ \theta -autoc}');xlabel('{t}')
            
            subplot(3,1,3)
            %             errorbar((1:len)'*paramObj.params.dt,nanmean(all_data_set_omega,2)...
            %                 ,nanstd(all_data_set_omega,0,2)*0.01,'LineWidth',1.5)
            plot((1:len)'*paramObj.params.dt,nanmean(all_data_set_omega,2)...
                ,'LineWidth',ld)
            hold on
            ylabel('{ \omega -autoc}');xlabel('{t}')
            %             paramObj.plot_pdf(all_data_set_omega)
            %             hold on
            %             xlabel('{ \omega -autoc}');ylabel('{PDF}')
        end
        
        function vec_padding = padding(obj,vec,len)
            vec = vec(:); % cols;
            if size(vec,1) < len
                vec_padding = cat(1,vec,NaN(len-size(vec,1),1));
            else
                vec_padding = vec(1:len,:);
            end
        end
        
        function get_distribution_spatial(obj)
        end
        function draw_spatial(obj)
        end
    end
    
    methods(Access = private)
        function y = autoc_vector(obj,x)
           M = size(x,1);
            temp = zeros(2*M-1,1);
            for cols = 1:size(x,2)
                %temp_v = xcorr(v(:,cols));
                temp =  temp + xcorr(x(:,cols),'unbiased');
            end 
            
            y = temp(M:end) / temp(M);
            
        end
    end
end

