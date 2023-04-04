classdef Basic < GetSingleParamsBase
    %BASIC Summary of this class goes here
    %   Detailed explanation goes here
    % velocity ,turn_angle, autocorrelation(normalized)
    properties
        flag = 0 % whether vector false;true
    end
    
    
    methods
        function obj = Basic(varargin)
            if nargin == 1
                obj.flag = varargin{1};
            end
        end
        
        
        function results = get_params(obj,tjObj)
            %Rg Construct an instance of this class
            % velocity
            % turn_angle
            % omega
            %v = [tjObj.xy(1:end-1,:)-tjObj.xy(2:end,:)]/tjObj.dt;
            v = [tjObj.xy(2:end,:)-tjObj.xy(1:end-1,:)]/tjObj.dt;
            %v = [mean(v(1:3,:),1);v];           
            %results.vec_v =  v;
            if obj.flag == 1
                results.v = v;
            else
                results.v = sqrt(sum(v.^2,2));
            end
                 results.v = obj.cal_val(tjObj,results.v);
                 
                 results.v_norm = obj.cal_val(tjObj,sqrt(sum(v.^2,2)));
            
            %results.v_norm = sqrt(sum(v.^2,2));
            cal_sgn = @(v11,v12,v21,v22) sign(det([v11, v12;v21,v22]));
%             sqrt(sum(v1,2))*sqrt(sum(v2,2))));
             sgn_val = arrayfun(@(v11,v12,v21,v22) cal_sgn(v11,v12,v21,v22),...
                 v(1:end-1,1),v(1:end-1,2),v(2:end,1),v(2:end,2));
             turn_angle =sgn_val.*abs(acos(sum(v(1:end-1,:) .* v(2:end,:),2)./...
                 (sqrt(sum(v(1:end-1,:).^2,2)) .*sqrt(sum(v(2:end,:).^2,2))))) ...
                 * (180 / pi); % cos
             
             results.turn_angle = obj.cal_val(tjObj,turn_angle);
%             results.turn_angle =arrayfun(@(v1,v2) cal_angle(v1,v2),v(1:end-1,:),v(2:end,:) );
            omega = (turn_angle(2:end) -  ...
                turn_angle(1:end-1))/ tjObj.dt;
            
            results.omega = obj.cal_val(tjObj,omega);
            
            
            %results.xy = obj.cal_val(tjObj,tjObj.xy-tjObj.xy(1,:));
            %\vec{v} dot
            %             M = size(v,1);
            %             autoc_x = xcorr(v(:,1)); autoc_y = xcorr(v(:,2));
            %             autoc = (autoc_x(M:end) + autoc_y(M:end)) /(autoc_x(M)+autoc_y(M));
            %             results.autoc = autoc;
            %plot(results.autoc)
        end
        
        function draw(obj,paramObj)
            % {{different num},{different num}}
            % ksdensity            
            %expression = ['paramObj.trajectory_params{m}','.',val];
            all_data_set_v = [];all_data_set_turn_angle = [];all_data_set_omega=[];
            for m = 1:numel(paramObj.trajectory_params)
                all_data_set_v = [all_data_set_v;...
                    paramObj.trajectory_params{m}.v_norm];
                all_data_set_turn_angle = [all_data_set_turn_angle;...
                    paramObj.trajectory_params{m}.turn_angle];
                all_data_set_omega = [all_data_set_omega;...
                    paramObj.trajectory_params{m}.omega];
                
            end
            %            index = find(all_data_set_alpha < 1);
            %            all_data_set_alpha(index) = [];
            %            all_data_set_Dt(index) = [];
            subplot(3,1,1)
            %paramObj.plot_log_pdf(all_data_set_alpha);
            paramObj.plot_log_pdf(all_data_set_v);
            hold on
            xlabel('|v|'); ylabel('{log PDF}'); 
            %title(['window size =' ,num2str(paramObj.params.window)])
            subplot(3,1,2)
            paramObj.plot_log_pdf(all_data_set_turn_angle)
            hold on
            xlabel('{\theta}');ylabel('{log PDF}')% legend(paramObj.params.legends)
            subplot(3,1,3)
            paramObj.plot_log_pdf(all_data_set_omega)
            hold on
            xlabel('{\omega}');ylabel('{log PDF}')% legend(paramObj.params.legends)
        end
        
        
        function y = cal_val(obj,tjObj,vec)
            
            if tjObj.window > 1
            cal_fun = @(vec) mean(vec{1},1);
            slices = tjObj.get_slices(vec);
            y =  cell2mat(arrayfun(@(vec) cal_fun(vec),slices,'UniformOutput',false));
            else
               y = vec; 
            end
        end
        
        function results = get_distribution_spatial(obj,tjObj,spatialObj)
            spatialObj.packed(tjObj.trajs,tjObj.trajectory_params,{'v','omega'});
            %spatialObj.packed(tjObj,tjObj.trajectory_params);
            results =  arrayfun(@(X) obj.get_distribution_spatial_step(spatialObj,X{:}),...
                spatialObj.X,'UniformOutput',false');
        end
        
        function results = get_distribution_spatial_step(obj,spatialObj,data)
            if obj.flag == 1
                vx =spatialObj.create_pos_val(data(:,1),data(:,2),data(:,3));
                vy = spatialObj.create_pos_val(data(:,1),data(:,2),data(:,4));
                v = (vx + vy);
%                 temp_v(isnan(temp_v)) = 0;
%                 results.v = temp_v;
                omega = spatialObj.create_pos_val(data(:,1),data(:,2),data(:,5));
%                 temp_omega = omega / omega(1);
%                 isnan(omega)
            else
                v =spatialObj.create_pos_val(data(:,1),data(:,2),data(:,3));
%                 temp_v = v/v(1);
               omega = spatialObj.create_pos_val(data(:,1),data(:,2),data(:,4));
%                 temp_
            end
            temp_v = v / v(1); temp_omega = omega / omega(1);
            temp_v(isnan(temp_v))=0; temp_omega(isnan(temp_omega)) = 0;
            %clear the NaN
            
            results.v = temp_v;
            results.omega = temp_omega;     
        end
        
        function draw_spatial(obj,spatialObj)
            v_set = [];
            v_set = [v_set;cell2mat( arrayfun(@(X) X{:}.v,...
                spatialObj.results,'UniformOutput',false'))];   
            omega_set = [];
            omega_set = [omega_set;cell2mat( arrayfun(@(X) X{:}.omega,...
                spatialObj.results,'UniformOutput',false'))];           
            %cell2mat(obj.results{:}.v) 
            subplot(2,1,1)
            %plot(spatialObj.len_index * spatialObj.dr,mean(v_set(2:end,:),1)','LineWidth',1.5)
            plot(mean(v_set(2:end,:),1)','LineWidth',1.5)
            hold on
            ylabel('{ v autoc_{spatial}}');xlabel('{\mu m}')
            subplot(2,1,2)
            %plot(spatialObj.len_index * spatialObj.dr ,mean(omega_set(3:end,:),1)','LineWidth',1.5)
            plot(mean(omega_set(3:end,:),1)','LineWidth',1.5)
            hold on
            ylabel('{ \omega autoc_{spatial}}');xlabel('{\mu m}')          
        end
        
        
        
        
        
        
    end
    

end

