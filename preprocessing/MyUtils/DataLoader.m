classdef DataLoader < handle
    properties
        % the data compose of (id,t,x,y,...,label) %id maybe loss
        path
        name_set=[];     
        threshold = 60;
        params% total num of trajectory
    end
    
    methods
        function obj = DataLoader(path,name_set,params)
            obj.path = path;
            obj.name_set = name_set;
            obj.params = params;%compose of (dt, baseline)
        end
        
        function new_trj = data_interp(obj,trj)
            % (t,x,y) single trajectory interp
            new_trj = [];
            trj = trj{1};
            if size(trj,1) < obj.threshold
                return ;
            end
            t = round((trj(:,1) - obj.params.baseline) / obj.params.dt);
            t_min = min(t); t_max = max(t);
            new_t = t_min:t_max;
            val_x = trj(:,2);
            val_y = trj(:,3);   
            x = interp1(t,val_x,new_t,'linear','extrap');
            y = interp1(t,val_y,new_t,'linear','extrap');       
            new_trj = [new_t',x',y']; 
        end
           
        function read_from_csv(obj,id_idx,lo,hi)
            % save the file in data    (lo:hi) = (t,x,y)
            data = {};
            
             if ~exist(obj.path,'dir')
                 mkdir(obj.path);
            end
            for name = obj.name_set
                name = name{1};
                filename = [obj.path , '/',name,'.csv']; 
                A = xlsread(filename);
                id = A(:,id_idx);
                data_set = A(:,lo:hi);
                rows = size(data_set,1);
                sorted_index  = unique(id);
                data_set = arrayfun(@(m) data_set(find(id == sorted_index(m)),:),...
                        1:length(sorted_index),'UniformOutput',false);
                    
                data_set = arrayfun(@(trj) obj.data_interp(trj), data_set','UniformOutput',false);
                data_set(cellfun(@isempty,data_set))=[];
                save([obj.path , '/',name,'.mat'],'data_set') ;
                disp([filename,' is loaded']);
            end
            %data(numel(data)+1 : numel(data) + rows) = data_set;         
        end
           
        function results = process(obj,varargin)
             
            data = {};
            index = [1];
            
            if nargin ==2 
                sampled = varargin{:};
            else
                sampled = 1;
            end
            
            for name = obj.name_set
                name = name{1};
                filename  = [obj.path , '/',name,'.mat'];
                 if exist(filename,'file') 
                    A = load(filename); 
                    A = A.data_set;
                    A_sampled = randperm(numel(A),round(numel(A) * sampled));
                    %A(A_sampled).
                    data = [data ; A(A_sampled)];
                    index = [index;numel(data)+1];
                 end
            end  
            
            results.data = data;
            results.index = index;
        end
        

        
    end
end

