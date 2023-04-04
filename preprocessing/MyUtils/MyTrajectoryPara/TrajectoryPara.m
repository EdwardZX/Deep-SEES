classdef TrajectoryPara < handle
    %the base of different param 
    % velocity omega turn_angle MSD(D_t, alpha)   
    properties
        xy; %(t,x,y)
        window;dt;index; get_params 
    end
    
    methods
        function obj = TrajectoryPara(X,params)
            %obj.X = X{1};
           
            obj.xy = X{1}(:,2:end);
            if  isnumeric(params.window) && params.window < size(obj.xy,1)
                obj.window = params.window;
            else
             obj.window = size(obj.xy,1); % full
            end
            %obj.window = params.window;
            obj.dt = params.dt;
            tau = obj.window-1;
            obj.index = (1:(size(obj.xy,1)-tau))' + [0,tau];
            obj.get_params = params.factory.get_params(obj);
            
        end
        
        function slices = get_slices(obj,varargin)
            
            if nargin == 2
               val = varargin{1};              
               tau = min(obj.window,size(val,1))-1;
               idx = (1:(size(val,1)-tau))' + [0,tau];
               slices = arrayfun(@(lo,hi) val(lo:hi,:),idx(:,1),idx(:,2),'UniformOutput',false');                             
            else
              slices = arrayfun(@(lo,hi) obj.xy(lo:hi,:),obj.index(:,1),obj.index(:,2),'UniformOutput',false');  
            end           
        end
    end
end