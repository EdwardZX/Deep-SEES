classdef TrajectorySet < handle
    %TRAJECTORYSET Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        trajs %cell format of {id, t, x, y,...}
        trajectory_params
        params; % window; dt;
    end
    
    methods
        function obj = TrajectorySet(trajs,params)
            %TRAJECTORYSET Construct an instance of this class
            %   Detailed explanation goes here
            obj.trajs = trajs;
            obj.params = params;
            %obj.trajectory_params = params.factory.get_params_factory(varargin);
        end
        
        function  process(obj,varargin)
            obj.trajectory_params =  arrayfun(@(X) TrajectoryPara(X,obj.params).get_params,...
                obj.trajs,'UniformOutput',false');      
            %arrayfun(obj.params.factory.get_params())
            %obj.trajectory_params = obj.trajectory_params.get_params(obj);
        end
        
        function show(obj)
            obj.params.factory.draw(obj)
        end
        
        function [f,idx] = plot_log_pdf(obj,x)
            %[f,xi] = ksdensity(x);
            % figure
            %histogram(x)
            %dx = abs(xi(2)-xi(1));
            %plot(xi,log(f*dx),'LineWidth',1.5)
            nbins = 50; sz = 20;                     
            %h=histogram(x,nbins,'Normalization','pdf'); 
            [pdf,edges] = histcounts(x,nbins, 'Normalization', 'probability');
            y = (edges(1:end-1) + edges(2:end))/2;
            idx = obj.get_peak_value(x,pdf,edges);
            
            %scatter(y,log10(pdf),sz)
            %plot(y,log10(pdf),'LineWidth',1)
            f = scatter(y,log10(pdf),sz,'o');
            
        end
        
        function [f,idx] = plot_pdf(obj,x)
%             [f,xi] = ksdensity(x);
%             dx = abs(xi(2) - xi(1));
%             f = plot(xi,f * dx,'LineWidth',1.5);
%             
%             
            nbins = 50; sz = 20;                     
            %h=histogram(x,nbins,'Normalization','pdf'); 
            [pdf,edges] = histcounts(x,nbins, 'Normalization', 'probability');
            y = (edges(1:end-1) + edges(2:end))/2;
            
            idx = obj.get_peak_value(x,pdf,edges);
            %scatter(y,log10(pdf),sz)
            %plot(y,log10(pdf),'LineWidth',1)
            f = scatter(y,pdf,sz,'o');
            %histogram(x,[0:0.1:2.5])
        end
        
        function idx = get_peak_value(obj,x,pdf,edges)
            [~,y_index] = max(pdf);
            idx = find(x >= edges(y_index) & x < edges(y_index+1));            
        end
        
         
        
    end
    

    
    
end

