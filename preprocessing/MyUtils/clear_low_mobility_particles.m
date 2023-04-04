function clear_low_mobility_particles(path,name)
params = struct('window','all','dt',0.017,'baseline',0.009);
factory = ParamsFactory();
sz = 30;
factory_temp = factory.get_params_factory({'msd',sz});
params.factory =  factory_temp;
dataloader = DataLoader(path,name,params);
dl = dataloader.process();
tj = TrajectorySet(dl.data,params);
tj.process();
index = [];
for m = 1: numel(tj.trajectory_params)
    if tj.trajectory_params{m}.alpha >= 2.4 ...
            || tj.trajectory_params{m}.alpha <= 1.25 ...
            || size(tj.trajs{m},1) < sz * 2
        index = [index;m];
    end

end
tj.trajs([index]) = [];
if ~exist('./data/clear/','dir')
    mkdir('./data/clear/')
end

data_set = tj.trajs;
save(['./data/clear/',name{:},'.mat'],'data_set')
end