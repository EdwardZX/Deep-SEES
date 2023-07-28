function get_data_2d()
%% path and filename
path = './data/';
save_path = './data/lstm_vae';
name_set = {'Test'};

%% parameters setting
moving_window = 30; % history length L
is_del_spatial_iso = 1; % is focus on local shape

%% brownian mask setting default
L_msd = 90; n_lags = 8; %default L_msd with 90 neighbors and tau = 8
params = struct('dt',1,'baseline',0,'window',L_msd);
factory = ParamsFactory();
params.factory =  factory.get_params_factory({'msd',n_lags});

%% running
for m = 1:numel(name_set)
    name = name_set(m);
    tj = get_value(path,name,params,'not_show'); %% if (xy, v ,theta) lost the first head
    if is_del_spatial_iso
        [t,xy,pos,res] = get_slice_t(tj,moving_window,is_del_spatial_iso);
    else
        [t,xy,pos] = get_slice_t(tj,moving_window,is_del_spatial_iso);
    end
    y_msd = get_slice_msd_value(tj,moving_window,params.window);
    
    data_set = cat_value(t,pos,xy);
    
    %% save
    if is_del_spatial_iso
        str_name = [name{:},'_iso'];
        csvwrite(fullfile(save_path,[str_name,'.csv']),data_set);
        data_set = cat_value(t,pos,y_msd,res);
        csvwrite(fullfile(save_path,[str_name,'_is_brownian.csv']),data_set);
    else
        str_name = name{:};
        csvwrite(fullfile(save_path,[str_name,'.csv']),data_set);
        data_set = cat_value(t,pos,y_msd);
        csvwrite(fullfile(save_path,[str_name,'_is_brownian.csv']),data_set);
    end
end

end

function tj = get_value(path,name,params,varargin)
dataloader = DataLoader(path,name,params);
dl = dataloader.process();
tj = TrajectorySet(dl.data,params);
tj.process();
if nargin == 3
    tj.show();
    drawnow;
end

end

function y = get_slice_value(tj,str,window,varargin)
if nargin == 4
    y = arrayfun(@(m) get_single_slices(tj.trajectory_params{m}.(str),window,...
        size(tj.trajs{m},1)-varargin{:}),...
        (1:numel(tj.trajectory_params))','UniformOutput',false);
else
    y = arrayfun(@(m) get_single_slices(tj.trajectory_params{m}.(str),window,...
        size(tj.trajs{m},1)),...
        (1:numel(tj.trajectory_params))','UniformOutput',false);
end
end
function [t,xy,pos,varargout] = get_slice_t(tj,window,is_del_spatial,varargin)

if is_del_spatial
    if nargin == 4
        [t,xy,pos,varargout{1}] = arrayfun(@(m) get_single_slices(tj.trajs{m},window,is_del_spatial,...
            size(tj.trajs{m},1)-varargin{:},1),... %size(tj.trajs{m},1)-varargin{:},1)
            (1:numel(tj.trajs))','UniformOutput',false');
    else
        [t,xy,pos,varargout{1}] = arrayfun(@(m) get_single_slices(tj.trajs{m},window,is_del_spatial,...
            size(tj.trajs{m},1),1),...
            (1:numel(tj.trajs))','UniformOutput',false');
    end
else
    if nargin == 4
        [t,xy,pos] = arrayfun(@(m) get_single_slices(tj.trajs{m},window,is_del_spatial,...
            size(tj.trajs{m},1)-varargin{:},1),... %size(tj.trajs{m},1)-varargin{:},1)
            (1:numel(tj.trajs))','UniformOutput',false');
    else
        [t,xy,pos] = arrayfun(@(m) get_single_slices(tj.trajs{m},window,is_del_spatial,...
            size(tj.trajs{m},1),1),...
            (1:numel(tj.trajs))','UniformOutput',false');
    end
    
    
end
end



function y= get_slice_msd_value(tj,window,msd_window,varargin)
y = arrayfun(@(m) (get_mean_single_slices_msd(tj.trajectory_params{m}.msd,window,...
    msd_window,varargin{:})),...
    (1:numel(tj.trajectory_params))','UniformOutput',false);

end



function result = get_mean_single_slices_msd(val,window,msd_window,varargin)
%% keep window < msd_window;
sz = size(val,1);
tau = window-1;
idx = (1-tau-1:sz)' + [0,tau];
%%  upper F/2-W-OFFSET; bottom F/2-W
padding_upper = floor(msd_window/2-window);
padding_bottom = ceil(msd_window/2-window);
%% upper
if padding_upper>0
    idx = [ones(padding_upper,size(idx,2));idx];
else
    idx(1:1-padding_upper,:) = [];
end
%%  bottom
if padding_bottom>0
    idx = [idx;sz.*ones(padding_upper,size(idx,2))];
else
    idx(end-padding_upper:end,:) = [];
end


idx(idx<1) = 1;
idx(idx > sz) = sz;



msd_mean = arrayfun(@(lo,hi) msd_fit_alpha_Dt(mean(val(lo:hi,:),1),varargin{:}),...
    idx(:,1),idx(:,2),'UniformOutput',false');
% y = arrayfun(@(x) x{:}.(str),...
%     msd_mean);
alpha = arrayfun(@(x) x{:}.alpha,...
    msd_mean);
y_error = arrayfun(@(x) x{:}.y_error,...
    msd_mean);
result = mat2cell([alpha,y_error],ones(size(alpha,1),1));
%% sz_b > window
end



function [slices,varargout] = get_single_slices(val,window,is_del_spatial,varargin)

tau = window-1;
idx = (1:(size(val,1)-tau))' + [0,tau];
if nargin == 5
    % special for t xy
    % t
    slices = arrayfun(@(lo,hi) max(val(lo:hi,varargin{2})),idx(:,1),idx(:,2));
    f_val = smooth_traj(val);
    
    if is_del_spatial
        varargout{1} = arrayfun(@(lo,hi) traj_rotation(val(lo:hi,2:3)-f_val(lo,2:3)),idx(:,1),idx(:,2),...
            'UniformOutput',false');
        varargout{2} = arrayfun(@(lo) f_val(lo,2:3),idx(:,1),...
            'UniformOutput',false');
    else
        varargout{1} = arrayfun(@(lo,hi) val(lo:hi,2:3),idx(:,1),idx(:,2),...
            'UniformOutput',false');
        varargout{2} = arrayfun(@(lo) val(lo,2:3),idx(:,1),...
            'UniformOutput',false');
    end
    % pos
    %         varargout{2} = arrayfun(@(hi) val(hi,2:3),idx(:,2),...
    %             'UniformOutput',false');
    
    % rotation
    if is_del_spatial
        varargout{3} = arrayfun(@(lo,hi) traj_rotation(val(lo:hi,2:3)-f_val(lo,2:3),2),idx(:,1),idx(:,2),...
            'UniformOutput',false');
    end
else
    slices = arrayfun(@(lo,hi) val(lo:hi,:),idx(:,1),idx(:,2),'UniformOutput',false');
end


end

function Y = smooth_traj(X)
p = 5e-2;
t = (1:size(X,1))';
Y = zeros(size(X));
for m = 1:size(X,2)
    Y(:,m) = csaps(t,X(:,m),p,t);
end
end

function xy = traj_rotation(xy,varargin)

f_xy = smooth_traj(xy);
p = f_xy(end,:)-f_xy(1,:);
pos = 2*(det([p;1,0])>0)-1;
theta = pos * acos(sum(p.*[1,0])/sqrt(sum(p.^2)));

T = [cos(theta),sin(theta);-sin(theta),cos(theta)];
xy = xy * T;

%
[~,idx_max] = max(abs(xy(:,2)));
flag = 2*(xy(idx_max,2)>0)-1;
xy(:,2) = flag*xy(:,2);


if nargin > 1
    xy = [theta,flag];
end

end



function df = cat_value(t,pos,varargin)
h=0;N = numel(varargin{1});
df= [];
for n = 1:N
    
    val_total=[];
    for k = 1:numel(varargin{1}{n})
        %k_max = numel(varargin{m}{n});
        val = [];
        for m = 1:numel(varargin)
            %for k = 1:numel(varargin{m}{n})
            temp = varargin{m}{n}{k};
            %% clear;
            %temp(isnan(temp)) = 0;
            % rows num
            h = max(size(temp,1),h);
            if ~isempty(val)
                val = [zeros(h-size(val,1),size(val,2));val];
                temp = [zeros(h-size(temp,1),size(temp,2));temp];
            end
            val = [val,temp];
        end
        val(isnan(val))=0;
        val = reshape(val,[],1)';
        val_total = [val_total;val];
    end
    cat_temp = [t{n},cell2mat(pos{n}),val_total];
    df = [df;cat_temp];
end
disp('cat finished')
end