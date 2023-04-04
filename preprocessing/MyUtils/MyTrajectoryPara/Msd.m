classdef Msd < GetSingleParamsBase
    %MSD Summary of this class goes here
    %   Detailed explanation goes here
    properties
        lag = 10;
    end
    
    methods
        function obj = Msd(lag)
            if nargin == 1
                obj.lag= lag;
            end
        end
        
        function results = get_params(obj,tjObj)
            %MSD Construct an instance of this class
            slices = tjObj.get_slices(); %for one trajectory
            temp =  cell2mat(arrayfun(@(vec) obj.msd(vec,tjObj),slices,...
                'UniformOutput',false'));
            
            results.alpha = temp(:,1);
            results.Dt = temp(:,2);
            results.msd = temp(:,4:end);
            results.y_error = temp(:,3);
%             t = repmat(1:obj.lag,size(results.msd,1),1);
% %             delta_x = slices(2:end,1) - slices(1:end-1,:);
%              results.delta_x_norm = sqrt(results.msd ./2 ./results.Dt ./t);
        end
        
        function result = msd(obj,vec,tjObj)
            %get the single slices msd
            % the len of single slice must be twice longer than lag
            vec = vec{1};
            num = size(vec,1) - obj.lag;
            getIndexM = @(len,tau) (1:len)' + [0,tau]; % (1:num-tau) ~ (1+tau : num)
            temp = zeros(obj.lag,1);
            for m = 1:obj.lag
                index = getIndexM(num,m);
                temp(m) = mean(sum((vec(index(:,1),:) - vec(index(:,2),:)).^2,2));
            end
            %expfit(temp)
            p = polyfit(log10((1:obj.lag)*tjObj.dt)',log10(temp),1);
            y_fit = polyval(p,log10((1:obj.lag))');
            %plot(log((1:obj.lag)*tjObj.dt)',log(temp))
            alpha = p(1);
            Dt = 10^p(2)/4;
            y_error = norm(log10(temp)-y_fit);
            result = [alpha, Dt,y_error,temp'];
        end
        
        function draw(obj,paramObj)
            % {{different num},{different num}}
            % ksdensity
            all_data_set_alpha = [];all_data_set_Dt = [];
            all_data_msd = [];
            % get_color
            
            %expression = ['paramObj.trajectory_params{m}','.',val];
            for m = 1:numel(paramObj.trajectory_params)
                % select
                all_data_set_alpha = [all_data_set_alpha;paramObj.trajectory_params{m}.alpha];
                all_data_set_Dt = [all_data_set_Dt;paramObj.trajectory_params{m}.Dt];
                all_data_msd = [all_data_msd;paramObj.trajectory_params{m}.msd];
                
            end
            %            index = find(all_data_set_alpha < 1);
            %            all_data_set_alpha(index) = [];
            %            all_data_set_Dt(index) = [];
            
            
            
            subplot(2,2,1)
            %paramObj.plot_log_pdf(all_data_set_alpha);
            [f,idx] = paramObj.plot_pdf(all_data_set_alpha);
            hold on
            xlabel('{\alpha}'); ylabel('{PDF}')
            subplot(2,2,2)
            paramObj.plot_log_pdf(all_data_set_Dt);
            hold on
            xlabel('{D_t}');ylabel('{log PDF}')% legend(paramObj.params.legends)
            
            
            
            subplot(2,2,3)
            
%             idx = intersect(idx_alpha,idx_Dt);
%             if isempty(idx)
%                 idx = idx_alpha;
%             end
            all_data_msd = all_data_msd(idx,:);
            per_m = randperm(size(all_data_msd,1),min(30,size(all_data_msd,1)));
            for m = per_m
%                 for j = 1:size(paramObj.trajectory_params{m}.msd,1)
%                     y = paramObj.trajectory_params{m}.msd;
%                     temp = y(j,:);
%                     t = 1:length(temp);
%                     plot(log10(t*paramObj.params.dt),log10(temp),'lineWidth',1.5,'Color',f.CData);
%                     hold on
%                 end
                            %temp = mean(paramObj.trajectory_params{m}.msd,1);
                            temp = all_data_msd(m,:);
                            t = 1:length(temp);
                            plot(log10(t*paramObj.params.dt),log10(temp),'lineWidth',0.3,'Color',f.CData);
                            hold on
            end
            xlabel('{log(t)}');ylabel('{log_{10} (Msd)}')
            
            
            subplot(2,2,4)
            %err = std(all_data_msd(m,:),1);
            plot(log10(t*paramObj.params.dt),log10(mean(all_data_msd,1)),'lineWidth',1.5);
            xlabel('{log(t)}');ylabel('{log_{10} (Msd)}')
            hold on
            
        end
        
        
        function get_distribution_spatial(obj)
        end
        
        function draw_spatial(obj)
        end
    end
end





