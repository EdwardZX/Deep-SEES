function transform_data(name,data,varargin)
data_set = cell(size(data,3),1); information=cell(size(data,3),1);
for m = 1:size(data,3)
    xy = data(:,:,m);
    data_set{m} = [(1:size(xy,1))',xy];
    if nargin > 2
    information{m} = varargin{1}(:,m);
    end
end

if nargin > 2
save(['./data/',name,'.mat'],'data_set','information');
else
save(['./data/',name,'.mat'],'data_set');   
end
end