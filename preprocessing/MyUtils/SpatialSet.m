classdef SpatialSet < handle
    %SPATIALSET Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        dr = 1;
        lo;hi
        X;% the region of the particles like 0,400
        results;
        params;
        len = 120; %200 um
        mem_index;
        len_index;
    end
    
    methods
        function obj = SpatialSet(params)
            %SPATIALSET Construct an instance of this class
            %   first concate the value of (xy,value),like velocity,...
            %obj.X = o;
            %obj.xRange = params.lo:params.dr:params.hi;
            %obj.yRange = params.lo:params.dr:params.hi;
            if nargin == 1
            obj.lo = params.lo; obj.hi = params.hi;
            obj.dr = params.dr;
            obj.params = params;
            %%circle index to sample
            obj.len = floor(obj.len/obj.dr);
            obj.mem_index =obj.get_circle_contour(obj.len);
            end
            
        end
        
        function contour_index = get_circle_contour(obj,len)
            A = zeros(len);
            for m = 1:len
                for n = 1:len
                    A(m,n) = sqrt((m-1)^2 + (n-1)^2);
                end
            end
            index = unique(A);
            contour_index = arrayfun(@(x) find(A == x), index,'UniformOutput',false');
            obj.len_index = index;
        end
        
        function  output = packed(obj,trajs,val,varargin)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            %             output = arrayfun(@(tj,x) obj.struct_value_packed(tj,x,names),...
            %                 trjObj.trajs,trjObj.trajectory_params,'UniformOutput',false');
            output = arrayfun(@(tj,x) obj.struct_value_packed(tj,x,varargin{:}),...
                trajs,val,'UniformOutput',false');
            
            output = obj.change_to_spatial(output);
            obj.X = output;
            %obj.X = obj.X(1:10);
        end
        
        function process(obj,tjObj)
            
            % obj
            obj.results = obj.params.factory.get_distribution_spatial(tjObj,obj);
            
            %             obj.results =  arrayfun(@(X) obj.get_distribution_spatial(X{:}),...
            %                 obj.X,'UniformOutput',false');
            
        end
        
        function new_vec = value_padding(obj,tj,vec)
            %in the middle of the xy position
            %vec = vec{:};
            % padding with 0
            header = ceil((size(tj,1) - size(vec,1)));
            new_vec = [zeros(header,size(vec,2));vec];
            %             t_min = min(t); t_max = max(t);
            %             new_t = t_min:t_max;
            %             vec = interp1(t(end-length(vec):end),vec,new_t,'linear','extrap');
            %new_vec = [tj,vec];
        end
        
        function new_vec = struct_value_packed(obj,tj,vec,varargin)
            tj = tj{:}; vec = vec{:};
            if isstruct(vec) && nargin ==4
                %names = fieldnames(vec);
                names = varargin{:};
                new_vec = [];
                for m = 1:numel(names)
                    new_vec = [new_vec,obj.value_padding(tj,vec.(names{m}))];
                end
                new_vec = [tj,new_vec];
                
            elseif isstruct(vec) && nargin <4
                names = fieldnames(vec);
                %names = varargin{:};
                new_vec = [];
                for m = 1:numel(names)
                    new_vec = [new_vec,obj.value_padding(tj,vec.(names{m}))];
                end
                new_vec = [tj,new_vec];                           
                %new_vec = [tj,obj.value_padding(tj,vec)];
                
            else
                new_vec = [tj,obj.value_padding(tj,vec)];
            end
            
        end
        
        function data = change_to_spatial(obj,data)
            %t(x,y,label) without id
            %data id(t,x,y) t = 0,1,2,3,... * dt
            temp = cell2mat(data);
            t_index = unique(temp(:,1));
            data = arrayfun(@(m) temp(find(temp(:,1) == t_index(m)),2:end),...
                1:length(t_index),'UniformOutput',false)';
        end
        
        function results = get_distribution_spatial(obj,data)
            %% cal_single_time_correlation
            %% omega and v
            %the data of the single time
            % create the range
            results.v =obj.create_pos_val(data(:,1),data(:,2),data(:,3));
            results.omega = obj.create_pos_val(data(:,1),data(:,2),data(:,4));
            
        end
        
        function result = create_pos_val(obj,x,y,val)
            x =  floor((x(:)- obj.lo)/obj.dr) + 1; %no 0
            y = floor((y(:)-obj.lo)/obj.dr) + 1;
            A = zeros(ceil(obj.hi-obj.lo)/obj.dr);
            A_cnt = ones(ceil(obj.hi-obj.lo)/obj.dr);
            for m = 1:size(x,1)
                %                 if x(m) < 1 || y(m)<1
                %                     disp('something wrong')
                %                 end
                A(x(m),y(m)) = A(x(m),y(m)) + val(m);
                A_cnt(x(m),y(m)) = A_cnt(x(m),y(m)) + 1;
            end
            A = A./A_cnt;
            M = ceil(obj.hi-obj.lo)/obj.dr;
            A = xcorr2(A);
            A = A(M:end,M:end); % to the center
            A = A(1:obj.len,1:obj.len);
            %len = round(210/obj.dr);
            result = arrayfun(@(m) mean(diag(rot90(A),m)), -obj.len + 1:1:0);
            %result = arrayfun(@(x) mean(A(x{:})), obj.mem_index)';
        end
        
        function show(obj)
            obj.params.factory.draw_spatial(obj)
        end
        
    end
end

