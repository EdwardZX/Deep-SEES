import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pickle
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection  import train_test_split
#from src.model import myDataSet,VAE_LSTM,my_scalar
from K_means_clustering import my_tsne,plot_graph_every_time,my_bar,data_processing,get_colors,my_scalar,plot_single_trajectory
from torch.utils.data import DataLoader, TensorDataset
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
from itertools import compress
import lstm_vae.utils as myutils
import os
from scipy import stats
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import copy



class VraeAnalyzer:

    # For filesets which all contain may trajectories
    # brownian_filter(delta,error)
    # ROI_select(lo,hi)
    # Reconstruct_error_select(lo,hi)
    # num_features, n_clustering
    # save_file_dir
    #

    # methods: tsne, hist,single_file_process, plot_filter_traj(add_raw,range),plot_kmeans_centers
    # kmeans: plot_filter_traj(add_raw,range),plot_kmeans_centers,


    def __init__(self,vrae,path,filename_set,num_features=2,seq_len=30,n_cluster=5,
                 brownian_filter=(0.1,0.05),alpha_ROI_select=(-np.inf,np.inf),Reconstruct_error_select=(0,2),
                 seed = 0,is_range = False ,n_scale = 3):
        self.num_features = num_features
        self.seq_len = seq_len

        self.n_cluster = n_cluster
        self.brownian_filter = brownian_filter
        self.alpha_ROI_select = alpha_ROI_select
        self.Reconstruct_error_select=Reconstruct_error_select
        self.vrae = vrae
        self.seed = seed
        self.is_range = is_range
        self.n_scale = n_scale

        self.iso = False
        self.xy_2D = False
        self.xy_3D = False
        self.is_save = False


        self.path = path
        self.filename_set = filename_set



        ##make dir
        # file_str = ''.join([i for i in self.filename_set[0] if not i.isdigit()])
        file_str = ''.join([i for i in self.filename_set[0]])
        file_str += '_multi' if len(filename_set)>1 else ''
        ## prefix
        # plt.savefig("./result/lstm_vae/" + file_str + '_total_error_rescontruct_from' +
        #             str(lo_idx) + '&&' + str(hi_idx) + '.png')
        self.result_path = ('./result/' + file_str + '/'
                        + 'clustering_num_'  + str(self.n_cluster)+ '/'
                       +'_brownian_filter_' + str(brownian_filter)
                       + '_alpha_ROI_select_' + str(alpha_ROI_select)
                       + '_Reconstruct_error_select_' + str(Reconstruct_error_select)
                       )
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
            os.makedirs(self.result_path + '/' + 'kmeans')

        ## get data and idx ...
        self._process()


    def _process(self):
        df, df_msd, idx = myutils.load_data(self.path,self.filename_set)
        ### get data and brownian filter
        c_output_total,z_latent_total,data_raw_total = self._run_single_file(df)
        z_latent = z_latent_total[1] # unormal 
        # z_latent = z_latent_total[1]
        z_norm = np.linalg.norm(z_latent_total[0], axis=1)
        if len(z_norm.shape) <= 1:
            z_norm = z_norm[:, None]

        idx[-1] = len(z_latent)
        self.filter_idx,self.non_filter_idx,brownian_filter = self._get_filtering_idx(df_msd,c_output_total[0],data_raw_total[0])
        z_latent[brownian_filter] = np.zeros_like(z_latent[brownian_filter])
        # z_latent[brownian_filter] = np.nan(z_latent[brownian_filter].shape[0],z_latent[brownian_filter].shape[1])
        self.raw = data_raw_total[0]
        self.z_latent = z_latent
        self.z_norm = z_norm
        self.output = c_output_total[0]
        self.idx = idx



        _,(self.t,self.pos) = myutils.data_processing(df,self.num_features,self.seq_len)
        self.T,self.mirror = self._get_res_rotation(df_msd)

        ####kmeans
        # load filter_z
        non_filter_idx = [i for i in range(len(self.z_latent))]
        non_brownian_filter_idx = list(set(non_filter_idx).difference(set(brownian_filter)))
        self.filter_z = z_latent[non_brownian_filter_idx]

        self.kmeans = KMeans(n_clusters=self.n_cluster, random_state=self.seed).fit(self.filter_z)

        # self.centers_labels = self.kmeans.labels_.copy()


        ###label###
        labels = -np.ones(len(z_latent),dtype=np.int8)
        labels[non_brownian_filter_idx] = self.kmeans.labels_
        ##filter
        labels[self.filter_idx] = -1
        self.usr_filter_labels = -1
        self.kmeans.labels_ = labels
        # self.usr_filter_labels = int(np.mean(self.kmeans.labels_[brownian_filter]))
        # self.kmeans.labels_[self.filter_idx] = -1
        # self.kmeans.labels_[self.kmeans.labels_ == self.usr_filter_labels] = -1

        ### cal_which node is -1




    def plot_filtering_traj(self,num=20,selected_idx=False,is_filtering=False, scalar=[1]):
        if len(scalar) ==1:
            scalar = [scalar[0] for i in range(self.num_features)]
        if selected_idx:
            idx = selected_idx
            sample_list = [i for i in range(len(idx))]
        else:
            idx = self.non_filter_idx
            np.random.seed(self.seed)
            sample_list = np.random.choice(range(len(idx)), min(num, len(idx)), replace=False)



        for i in sample_list:  # sample_list:#range(0,len(idx_reconstruct),steps):#range(0,int(0.05 * reconstruct_error.shape[0]),100):
            if not is_filtering:
                plt.plot(self.raw[idx[i], :, 0]/scalar[0],
                     self.raw[idx[i], :, 1]/scalar[1],
                     c='b')
            plt.plot(self.output[idx[i], :, 0]/scalar[0],
                     self.output[idx[i], :, 1]/scalar[1],
                     c='orange')
            # plt.show()

        plt.axis("equal")
        plt.savefig(self.result_path + '/' + 'filtering_traj.pdf',format="pdf",transparent=True)
        # plt.savefig(self.result_path + '/' + 'filtering_traj.eps', dpi=600, format='eps')
        plt.show()

    def plot_clustering_traj(self, num=20, selected_idx=False, scalar=[1]):
        if len(scalar) ==1:
            scalar = [scalar[0] for i in range(self.num_features)]
        if selected_idx:
            idx = selected_idx
            sample_list = [i for i in range(len(idx))]
        else:
            idx = self.non_filter_idx
            np.random.seed(self.seed)
            sample_list = np.random.choice(range(len(idx)), min(num, len(idx)), replace=False)

        color_set, cluster_index = myutils.get_colors(self.kmeans.cluster_centers_)
        lb_centers = set(self.kmeans.labels_)

        for i in sample_list:  # sample_list:#range(0,len(idx_reconstruct),steps):#range(0,int(0.05 * reconstruct_error.shape[0]),100):
            if self.kmeans.labels_[idx[i]] == -1:
                continue

            plt.plot(self.output[idx[i], :, 0]/scalar[0],
                     self.output[idx[i], :, 1]/scalar[1],
                     alpha=0.5,
                     c=color_set[int(cluster_index[self.kmeans.labels_[idx[i]]]), :])
            # plt.show()

        plt.axis("equal")
        plt.savefig(self.result_path + '/' + 'clustering_traj.pdf',format="pdf",transparent=True)
        # plt.savefig(self.result_path + '/' + 'clustering_traj.eps',dpi=600, format='eps')
        plt.show()



    def plot_filtering_AC(self,num=5):



        idx = self.non_filter_idx
        np.random.seed(self.seed)
        sample_list = np.random.choice(range(len(idx)), min(num, len(idx)), replace=False)

        cols = self.raw.shape[2]


        for i in sample_list:  # sample_list:#range(0,len(idx_reconstruct),steps):#range(0,int(0.05 * reconstruct_error.shape[0]),100):
            for j in range(self.raw.shape[2]):


                plt.subplot(cols, 1,j+1)
                plt.plot(self._autocorr(self.raw[idx[i], :, j]),
                     c='b')
                plt.plot(self._autocorr(self.output[idx[i], :, j]),
                         c='orange')
                # plt.plot(self._autocorr(self.raw[idx[i], :, j]-self.output[idx[i], :, j]),
                #          c='b')



        plt.savefig(self.result_path + '/' + 'AC_traj.pdf',format='pdf',Transparent=True)
        # plt.savefig(self.result_path + '/' + 'AC_traj.eps',dpi=600, format='eps')
        plt.show()



    def _autocorr(self,x):
        result = np.correlate(x, x, mode='full')
        return result[result.size // 2:]

    def run_centers(self,scalar=1):
        kmeans = self.kmeans
        vrae = self.vrae
        z_norm = self.z_norm

        centers = copy.deepcopy(kmeans.cluster_centers_)
        # print(centers[0])
        label = kmeans.labels_


        batch_size = vrae.batch_size
        color_set, cluster_index = myutils.get_colors(centers)
        # centers_batch = centers[np.newaxis,:].repeat(batch_size,axis=0)
        centers_batch = centers[np.newaxis, :].repeat(batch_size, axis=0)
        if not ((np.linalg.norm(centers, axis=1) < 1.02).any() and (np.linalg.norm(centers, axis=1) > 0.98).any()):
            print('normalized')
            for i in range(len(centers)):
                ###
                centers[i] = centers[i] * np.mean(z_norm[label == i, :])
        centers_batch = centers.repeat(batch_size, axis=0)


        np.random.seed(0)
        # print(centers_batch.shape)
        data = centers_batch + 1e-2 * np.random.randn(centers_batch.shape[0], centers_batch.shape[1])
        dataset = TensorDataset(torch.from_numpy(data))

        decoded = vrae.latent2decoded(dataset).transpose(1, 0, 2) /scalar

        if self.num_features == 3:
            fig = plt.figure(figsize=(6, 5))
            ax = fig.add_subplot(111, projection='3d')
            for i in range(len(centers)):
                if i == self.usr_filter_labels:
                 continue
                for dataset in decoded[i * batch_size:(i + 1) * batch_size, :,:]:
                    ax.plot(dataset[:,0],dataset[:,1],dataset[:,2],
                        '-',
                        alpha=0.5,
                        c=color_set[int(cluster_index[i])],
                        zorder=0,
                        linewidth=0.5)
                 # plt.plot(decoded[i * batch_size:(i + 1) * batch_size, :, 0].T,
                 #     decoded[i * batch_size:(i + 1) * batch_size, :, 1].T, c=color_set[int(cluster_index[i])])
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            ax.set_axis_off()
            ax.grid(False)
        else:
            for i in range(len(centers)):
                if i == self.usr_filter_labels:
                    continue
                plt.plot(decoded[i * batch_size:(i + 1) * batch_size, :, 0].T,
                         decoded[i * batch_size:(i + 1) * batch_size, :, 1].T, c=color_set[int(cluster_index[i])])

                plt.axis("equal")
            # plt.show()
        # plt.axis('equal')
        # ax = plt.gca()
        # ax.set_aspect('equal', 'box')
        # plt.axis("equal")
        plt.savefig(self.result_path + '/'+str(len(centers))+'run_centers.pdf', format="pdf",transparent=True)
        # print(decoded.shape)
        np.savetxt(self.result_path + '/' +str(len(centers))+'output_centers.csv',
                   decoded.reshape(decoded.shape[0],-1),
                   delimiter=',', fmt='%10.8f')
        
        plt.show()

    def find_kmeans_k(self):
        # elbow methods
        SSE = []
        SIL = []#silhouette
        for i  in range(2,15):
            estimator = KMeans(n_clusters=i, random_state=self.seed).fit(self.filter_z)
            SSE.append(estimator.inertia_) # elbow methods
            SIL.append(metrics.silhouette_score(self.filter_z,estimator.labels_))
            print(i)

        t = [i for i  in range(2,15)]
        plt.xlabel('k')
        plt.ylabel('SSE')
        plt.plot(t,SSE, 'o-')
        plt.show()
        plt.xlabel('k')
        plt.ylabel('silhouette')
        plt.plot(t,SIL,'+-')
        plt.show()

    def re_clustering(self,id_k,num_clustering=5):
        z_latent_select = self.z_latent[self.kmeans.labels_ == id_k]
        re_kmeans = KMeans(n_clusters=num_clustering, random_state=self.seed).fit(z_latent_select)
        labels = -np.ones_like(self.kmeans.labels_)
        labels[self.kmeans.labels_ == id_k]= re_kmeans.labels_
        self.kmeans = re_kmeans
        self.kmeans.labels_ = labels
        self.is_save = False

    def re_multilayers_train(self,id_k,num_clustering):
        df, df_msd, idx = myutils.load_data(self.path, self.filename_set)
        re_id = np.where(self.kmeans.labels_ == id_k)[0]
        data, _= myutils.data_processing(df.iloc[re_id], self.num_features, self.seq_len)
        if self.is_range:
            self.scalar = self.n_scale / np.mean(np.mean(np.abs(data), axis=0), axis=0)
        else:
            scalar = self.n_scale / np.mean(np.mean(np.abs(data)))
            self.scalar = [scalar for i in range(self.num_features)]
        data = data * self.scalar

        vrae = self.vrae
        dataset = TensorDataset(torch.from_numpy(data))
        vrae.fit(dataset)
        vrae.eval()
        _, z_latent_total, _ = self._run_single_file(df)
        z_latent = z_latent_total[1]  # unormal
        z_latent_select = z_latent[re_id]
        re_kmeans = KMeans(n_clusters=num_clustering, random_state=self.seed).fit(z_latent_select)
        labels = -np.ones_like(self.kmeans.labels_)
        labels[re_id] = re_kmeans.labels_
        self.kmeans = re_kmeans
        self.kmeans.labels_ = labels
        self.is_save = False

    def plot_every_hist(self):
        for i in range(len(self.filename_set)):
            z, labels, centers,df_traj = self._load_each_file(i)
            val = myutils.my_bar(labels, centers)
            plt.savefig(self.result_path + '/' + 'histogram' + self.filename_set[i] + '.pdf',format='pdf',Transparent=True)
            plt.show()

    def plot_every_tsne(self,sampled = 0.1):
        for i in range(len(self.filename_set)):
            (z,labels), _, centers,_ = self._load_each_file(i)
            _, x_test, _, y_test = train_test_split(z, labels, test_size=sampled, random_state=0)
            myutils.my_tsne(x_test, y_test, centers)

            plt.savefig(self.result_path + '/' + 'tsne' + self.filename_set[i] + '.pdf',format='pdf',Transparent=True)
            plt.show()

    def plot_every_trajectory(self,is_filtering=True,lo=0,step=-1,dim='2d',sz=0.05, is_equal =True):
        for i in range(len(self.filename_set)):
            z, labels, centers,(xy,t) = self._load_each_file(i,is_filtering)

            if self.num_features == 3:
                myutils.plot_single_trajectory_3d(xy, t,labels, centers, lo = lo,steps_set=step)
            elif self.num_features == 3 and self.iso and self.xy_2D:
                myutils.plot_single_trajectory(xy, t, labels, centers,lo=lo,steps_set=step,sz=sz,is_equal = is_equal)
                if is_equal:
                    plt.axis('equal')
            else:
                myutils.plot_single_trajectory(xy, t, labels, centers, lo=lo, steps_set=step, sz=sz, is_equal = is_equal)
                if is_equal:
                    plt.axis('equal')

            # plt.xlim([50,70])
            # plt.ylim([0,20])
            # str_filtering = '_is_filtering_' if is_filtering else ''
            # str_3d = '_is_3d_' if dim=='3d' else ''

            plt.savefig(self.result_path + '/' + 'trajectory_colors_' +
                        self.filename_set[i] +
                        '_range_'+str(lo)+'_step_'+str(step)+
                        # str_filtering +str_3d+
                        '.pdf', format='pdf', Transparent = True) #the name is so long
            plt.show()

    def plot_every_segement(self,lo=0,step_set=-1, labels_flag=False):
        for i in range(len(self.filename_set)):
            z, labels, centers,(xy,t) = self._load_each_file(i,is_filtering=True)
            color_set, cluster_index = myutils.get_colors(centers)

            steps = len(xy)
            if step_set > 0 and step_set + lo < len(xy):
                steps = step_set

            lb_centers = set(labels)
            out_mat = np.tile(labels[lo:min(lo + steps, labels.shape[0])], (int(steps / 10), 1))
            if labels_flag:
                # masked_array = np.ma.masked_where(out_mat == -1, out_mat)
                masked_array = np.ma.masked_where(out_mat != labels_flag, out_mat)
            else:
                masked_array = np.ma.masked_where(out_mat == -1, out_mat)
            # masked_array = np.ma.masked_where(out_mat == -1, out_mat)  
            cmap = plt.cm.get_cmap('rainbow',len(centers))
            # print(len(centers))
            # cmap = cm.rainbow(np.linspace(0, 1, len(lb_centers)))
            cmap.set_bad(color='#D3D3D3')
            plt.imshow(masked_array, cmap=cmap, vmin=0, vmax=len(centers) - 1)
            plt.colorbar()

            plt.savefig(self.result_path + '/' + 'trajectory_segment_' +
                        self.filename_set[i] +
                        '_range_'+str(lo)+'_step_'+str(steps)+
                        # str_filtering +str_3d+
                        '.pdf',
                        format='pdf',Transparent=True) #the name is so long
            plt.show()


    def plot_every_time_series(self, lo=0, step=-1, n_dim=3,extend=0.2,sz=0.005):
        for i in range(len(self.filename_set)):
            z, labels, centers, (xy, t) = self._load_each_file(i, True)
            myutils.plot_single_time_series(xy, t, labels, centers, lo=lo, steps_set=step,
                                            n_dim=n_dim,extend_para=extend,sz=sz)

            plt.savefig(self.result_path + '/' + 'time_series_colors_' +
                        self.filename_set[i] +
                        '_range_' + str(lo) + '_step_' + str(step) +
                        '.pdf',
                        format='pdf',Transparent=True)  # the name is so long
            plt.show()


    def plot_filtering_traj_overlay(self,lo=0,step=-1):
        for i in range(len(self.filename_set)):
            _, labels, centers,(raw,t) = self._load_each_file(i,is_filtering=False)
            _, labels, centers, (output, t) = self._load_each_file(i, is_filtering=True)
            if step == -1:
                plt.plot(raw[:, 0],
                         raw[:, 1],
                         c='b')
                plt.plot(output[:, 0],
                     output[:, 1],
                     c='orange')
            else:
                plt.plot(raw[lo:lo+step, 0],
                         raw[lo:lo+step, 1],
                         c='b')
                plt.plot(output[lo:lo+step,0],
                         output[lo:lo+step,1],
                         c='orange')

            plt.axis('equal')
            plt.savefig(self.result_path + '/' + 'filtering_overlay' +
                        self.filename_set[i] +
                        '.pdf',format='pdf',Transparent=True) #the name is so long
            plt.show()



    def plot_every_multi_points(self, is_filtering=False):
        df, df_msd, idx = myutils.load_data(self.path, self.filename_set)
        for i in range(len(self.filename_set)):
            z, labels, centers,(pos,_) = self._load_each_file(i,is_filtering)

            df_kmeans = pd.concat([df.iloc[idx[i]:idx[i + 1], 0], pd.DataFrame(pos), pd.DataFrame(z), pd.DataFrame(labels)],
                                    axis=1)
            save_pth = self.result_path +'/kmeans/'+ self.filename_set[i] + '/'
            if not os.path.exists(save_pth):
                os.makedirs(save_pth)
            myutils.plot_graph_every_time(df_kmeans, centers, save_pth)

        print('finished')


    def _load_each_file(self,i,is_filtering=False):
        idx = self.idx
        z = self.z_latent[self.idx[i]:self.idx[i + 1], :]
        labels = self.kmeans.labels_[self.idx[i]:self.idx[i + 1]]
        centers = self.kmeans.cluster_centers_

        p = self.output.shape[1]
        id_p = int(p / 2)

        if self.num_features ==3 and self.iso and self.xy_3D:
            T_y = self.T[0][idx[i]:idx[i + 1], :]
            T_z = self.T[1][idx[i]:idx[i + 1], :]

            mirror_y = self.mirror[0][idx[i]:idx[i + 1]]
            mirror_z = self.mirror[1][idx[i]:idx[i + 1]]
            T = (T_y,T_z)
            mirror = (mirror_y,mirror_z)
        else:
            T = self.T[idx[i]:idx[i + 1], :]
            mirror = self.mirror[idx[i]:idx[i + 1]]

        if is_filtering:
            xy = self.output[idx[i]:idx[i + 1],:,:]/self.scalar
            # xy = self._reverse_mean_and_error(self.output[idx[i]:idx[i + 1], :, :] / self.scalar)
            # for a in [52800,52830,52860]:
            #     idx_temp1 = a
            #     idx_temp2 = a+3
            #     T1 = self.T[idx_temp1,:]
            #     T2 = self.T[idx_temp2, :]
            #     mirror1 = self.mirror[idx_temp1]
            #     mirror2 = self.mirror[idx_temp2]
            #     xy1 = self._xy_reverse_rotation(self.raw[idx_temp1, :, :]/self.scalar,T=T1,mirror=mirror1)
            #     xy2 = self._xy_reverse_rotation(self.raw[idx_temp2,:,:]/self.scalar,T=T2,mirror=mirror2)
            #     output1 = self._xy_reverse_rotation(self.output[idx_temp1, :, :] / self.scalar,T=T1,mirror=mirror1)
            #     output2 = self._xy_reverse_rotation(self.output[idx_temp2, :, :]/self.scalar,T=T2,mirror=mirror2)
            #
            #     plt.plot(xy1[:,0] + self.pos[idx_temp1, 0],
            #              xy1[:,1] + self.pos[idx_temp1, 1])
            #
            #     plt.plot(output1[:,0]+self.pos[idx_temp1,0],
            #              output1[:,1]+self.pos[idx_temp1,1])
            #
            #     plt.plot(xy2[:, 0] + self.pos[idx_temp2, 0],
            #              xy2[:, 1] + self.pos[idx_temp2, 1])
            #
            #     plt.plot(output2[:, 0] + self.pos[idx_temp2, 0],
            #              output2[:, 1] + self.pos[idx_temp2, 1])
            #     plt.show()
            # print('a')
            # tanspose
        else:
            xy = self.raw[idx[i]:idx[i + 1],:,:]/self.scalar

        for j in range(xy.shape[0]):
            if self.num_features==3 and self.iso and self.xy_3D:
                xy[j, :] = self._xy_reverse_rotation(xy[j, :], (T[0][j, :], T[1][j, :]),
                                                     (mirror[0][j], mirror[1][j]))
            else:
                xy[j, :] = self._xy_reverse_rotation(xy[j, :], T[j, :], mirror[j])

        if self.iso:
            xy += self.pos[idx[i]:idx[i + 1], :][:,None,:]


        ########cal t #######
        t = self.t[idx[i]:idx[i + 1]]

        delta_t = np.hstack([1, t[1:] - t[0:-1]])
        filter_t = (delta_t != 1)
        delta_t = list(compress(range(len(filter_t)), filter_t))
        delta_t = np.array(delta_t)
        delta_t = delta_t[delta_t < len(xy)]

        delta_t = np.hstack([0, delta_t, len(xy)])

        # xy_total = np.array([])

        ### similarity
        # similarity = metrics.pairwise.paired_distances(z[0:-1,:],z[1:,:])
        # z_similarity = np.insert(similarity,0,np.mean(similarity[0:3]))
        # labels_total = np.array([])
        if not len(delta_t):
            xy_total,labels_total = self._reverse_mean_and_error(xy,labels)
            t_total = np.hstack(
                        [np.linspace(t-self.seq_len,t,1),t])

        else:
            t_idx_delete = []
            for j in range(len(delta_t)-1):
                t_idx_delete.append(int(delta_t[j + 1])-1)
                if j == 0:
                    xy_total, labels_total = self._reverse_mean_and_error(xy[int(delta_t[j]):int(delta_t[j + 1]), :, :],
                                                                        labels[int(delta_t[j]):int(delta_t[j + 1])])
                    t_total = np.hstack(
                        [np.arange(t[int(delta_t[j])]-self.seq_len+1,t[int(delta_t[j])],1),t[int(delta_t[j]):int(delta_t[j + 1])]])
                    ##reverse
                else:
                    xy_temp,labels_temp = self._reverse_mean_and_error(xy[int(delta_t[j]):int(delta_t[j + 1]), :,:],
                                                                 labels[int(delta_t[j]):int(delta_t[j + 1])]) ##reverse
                    t_temp = np.hstack(
                        [np.arange(t[int(delta_t[j])]-self.seq_len+1,t[int(delta_t[j])],1),t[int(delta_t[j]):int(delta_t[j + 1])]])
                    xy_total = np.vstack([xy_total,xy_temp])
                    labels_total = np.hstack([labels_total,labels_temp])
                    t_total = np.hstack([t_total,t_temp])
            # print(len(t_idx_delete))
            # t = np.delete(t,t_idx_delete,0)
            # z_similarity = np.delete(z_similarity,t_idx_delete,0)
        ####### similarity of z-latent #####
        # print(z_similarity.shape, xy_total.shape, t.shape)

        if (not self.is_save) and is_filtering:
            np.savetxt(self.result_path + '/' + 'output.csv',
                   xy_total,
                   delimiter=',', fmt='%10.8f')
            np.savetxt(self.result_path + '/' + 'output_similarity.csv',
                   np.hstack((t.reshape(-1,1),z)),
                   delimiter=',', fmt='%10.8f')
            np.savetxt(self.result_path + '/' + 'output_label.csv',
                       labels_total,
                       delimiter=',', fmt='%d')
            np.savetxt(self.result_path + '/' + 'output_txy.csv',
                       np.hstack((t_total.reshape(-1,1),xy_total)),
                       delimiter=',', fmt='%10.8f')
            self.is_save = True
        return (z,labels), labels_total, centers,(xy_total,t_total)  #labels


    def _run_single_file(self,df):
        #return (output,z,raw) with normalize
        data,_ = myutils.data_processing(df,self.num_features,self.seq_len)
        if self.is_range:
            self.scalar = self.n_scale / np.mean(np.mean(np.abs(data), axis=0), axis=0)
        else:
            scalar = self.n_scale / np.mean(np.mean(np.abs(data)))
            self.scalar = [scalar for i in range(self.num_features)]
        data = data * self.scalar


        vrae = self.vrae
        dataset = TensorDataset(torch.from_numpy(data))
        output = vrae.reconstruct(dataset).transpose(1, 0, 2)
        z_run = vrae.transform(dataset)

        # print(data.shape,output.shape)

        ##normalization |v|/||v||
        z_norm = np.linalg.norm(z_run, axis=1)
        z_norm = z_run / z_norm[:, None]
        output_norm = np.linalg.norm(output, axis=1)
        output_norm = output / output_norm[:, None]

        data = data[:, :, :]
        data_norm = np.linalg.norm(data, axis=1)
        data_norm = data / data_norm[:, None]

        # D =  np.sqrt(((data[:z_norm.shape[0],:,:].reshape(z_norm.shape[0],-1) 
        #       - output.reshape(z_norm.shape[0],-1)
        # )**2).mean(axis=1))
        # print(D.shape)

        # np.savetxt(self.result_path + '/' + 'raw_output.csv',
        #            output.reshape(output.shape[0],-1)/self.scalar,
        #            delimiter=',', fmt='%10.8f')
        # np.savetxt(self.result_path + '/' + 'z_latent.csv',
        #            z_run,
        #            delimiter=',', fmt='%10.8f')
        # np.savetxt(self.result_path + '/' + 'D.csv',
        #            D,
        #            delimiter=',', fmt='%10.8f')

        return (output, output_norm), (z_run, z_norm), (data, data_norm) #

    def _get_brownian_filter(self,df_msd):
        alpha = df_msd.iloc[:, self.num_features+1].values
        y_error = df_msd.iloc[:, self.num_features+2].values

        plt.hist(alpha, bins=200)
        plt.title('alpha')
        plt.show()

        plt.hist(y_error, bins=200)

        plt.title('polynomial fitting error')
        plt.show()
        ######alpha 0.9-1.1 think it's quiet brownian
        idx_alpha = (alpha > 1 - self.brownian_filter[0]) & (alpha < 1 + self.brownian_filter[0])
        idx_model_error = y_error < self.brownian_filter[1]

        filter_idx = (idx_alpha & idx_model_error)  # think it's brownian need to filter

        return list(compress(range(len(filter_idx)), filter_idx))

    def _get_res_rotation(self,df_msd):
        # t pos(x num_features)
        df_msd_temp = df_msd.iloc[:, self.num_features+3:]

        if df_msd_temp.shape[1] ==2 : #2 dimensions
            theta = -df_msd_temp.iloc[:, 0].values #revers_theta
            mirror = df_msd_temp.iloc[:, 1].values
            T = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]]).transpose((2,0,1))
            self.iso = True
            self.xy_2D = True
        elif df_msd_temp.shape[1] == 4: #3 dimensions
            theta = -df_msd_temp.iloc[:, 0].values
            phi = -df_msd_temp.iloc[:, 1].values# revers_theta
            mirror_y = df_msd_temp.iloc[:, 2].values
            mirror_z = df_msd_temp.iloc[:, 3].values
            T_theta = np.array([[np.cos(theta), np.sin(theta),np.zeros([df_msd_temp.shape[0]])],
                                [-np.sin(theta), np.cos(theta),
                                 np.zeros([df_msd_temp.shape[0]])],
                                [np.zeros([df_msd_temp.shape[0]]),np.zeros([df_msd_temp.shape[0]]),np.ones([df_msd_temp.shape[0]])]]).transpose((2, 0, 1))

            T_phi = np.array([[np.cos(phi), np.zeros([df_msd_temp.shape[0]]),np.sin(phi)],
                                [np.zeros([df_msd_temp.shape[0]]), np.ones([df_msd_temp.shape[0]]),
                                 np.zeros([df_msd_temp.shape[0]])],
                              [-np.sin(phi),np.zeros([df_msd_temp.shape[0]]),np.cos(phi)]]).transpose((2, 0, 1))
            T = (T_theta,T_phi)
            mirror = (mirror_y,mirror_z)
            self.iso = True
            self.xy_3D = True
        else:
            theta = np.zeros([df_msd_temp.shape[0]])
            T = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]).transpose((2, 0, 1))
            mirror = np.ones([df_msd_temp.shape[0]])

        return T,mirror

    def _xy_reverse_rotation(self,xy,T,mirror):
        # xy: 30xnum_feature T:2x2

        if self.xy_3D and self.iso: #self.num_features==3
            if len(xy.shape) == 1:
                xy[1] *= mirror
            else:
                xy[:, 1] *= mirror[0]  # y
                xy[:, 2] *= mirror[1]  # z

            answer = np.dot(np.dot(xy, T[1]), T[0])
        elif self.xy_2D : #self.num_features==2
            if len(xy.shape) == 1:
                xy[1] *= mirror
            else:
                xy[:, 1] *= mirror # xy[:, 1] *= mirror
            xy[:,:2]  =  np.dot(xy[:,:2], T)
            answer = xy
        else:
            answer = xy

        return answer





    def _get_alpha_ROI_select_filter(self,df_msd):
        alpha = df_msd.iloc[:, self.num_features+1].values
        idx_alpha_filter = (alpha < self.alpha_ROI_select[0]) | (alpha > self.alpha_ROI_select[1])
        return list(compress(range(len(idx_alpha_filter)), idx_alpha_filter))

    def _get_reconstruction_error_filter(self,output,raw):
        # use_vi v_j/||vi|||v_j||| as dist
        # MSE
        sz = output.shape[0]
        raw = raw[:sz, :]
        output = output #- output[:, 0, :][:, None, :]  ##add OFFSET
        raw = raw #- raw[:, 0, :][:, None, :]

        output_norm =  np.linalg.norm(output-output.mean(axis=1)[:,None,:], axis=1)
        raw_norm = np.linalg.norm(raw-raw.mean(axis=1)[:,None,:], axis=1)
        # mse_error_norm = 1 - (((output * raw)).sum(axis=1) / np.linalg.norm(output, axis=1)
        #                       / np.linalg.norm(raw, axis=1))

        mse_error_norm = (((output-raw)**2).sum(axis=1) / output_norm
                              / raw_norm)
        mse_error_norm[np.isnan(mse_error_norm)] = np.nanmax(mse_error_norm)
        mse_error_norm = np.linalg.norm(mse_error_norm, axis=1) / np.sqrt(output.shape[2])
        # mse_error_norm = mse_error_norm/max(mse_error_norm)
        reconstruct_index = mse_error_norm.argsort()  #

        idx_lo = reconstruct_index[:int(self.Reconstruct_error_select[0]*reconstruct_index.shape[0])]
        idx_hi = reconstruct_index[int(self.Reconstruct_error_select[1]*reconstruct_index.shape[0]):]
        # idx = (mse_error_norm > self.Reconstruct_error_select[1]) | (mse_error_norm < self.Reconstruct_error_select[0])
        # idx = list(compress(range(len(idx)), idx))

        # plt.hist(mse_error_norm, bins=200)
        # plt.show()

        return mse_error_norm, np.hstack([idx_lo,idx_hi]) #idx

    def _get_filtering_idx(self,df_msd,output,raw):
        reconstruct_error, filter_idx = self._get_reconstruction_error_filter(output, raw)

        brownian_filter = self._get_brownian_filter(df_msd[:len(output)])
        roi_filter = self._get_alpha_ROI_select_filter(df_msd[:len(output)])

        filter_idx = list(set(brownian_filter).union(set(filter_idx)))
        filter_idx = list(set(roi_filter).union(set(filter_idx)))

        non_filter_idx = [i for i in range(len(output))]
        non_filter_idx = list(set(non_filter_idx).difference(set(filter_idx)))

        return filter_idx, non_filter_idx,brownian_filter

    def _reverse_mean_and_error(self,xy,reverse_labels):

        ######reverse window to filter pos####
        ###### output_off_xy
        pos_label = np.linspace(-6,6,31)
        norm_label_kernel = np.exp(-(pos_label) ** 2 / 2)
        norm_label_kernel = np.ceil(norm_label_kernel / np.exp(-(2.5) ** 2 / 2)).astype(int)[1:]
        norm_label_weight = np.ones(xy.shape[1],dtype='int64')
        tau = np.min([norm_label_kernel.shape[0]//2, norm_label_weight.shape[0]//2])
        res = np.min([norm_label_kernel.shape[0] - tau, norm_label_weight.shape[0]-tau])

        norm_label_weight[norm_label_weight.shape[0]//2 - tau: norm_label_weight.shape[0]//2+res] = \
        norm_label_kernel[norm_label_kernel.shape[0]//2 - tau: norm_label_kernel.shape[0]//2+res]

        if self.num_features == 3:
            raw_len = xy.shape[1] - 1 + xy.shape[0]
            reverse_list_x = [[] for i in range(raw_len)]
            reverse_list_y = [[] for i in range(raw_len)]
            reverse_list_z = [[] for i in range(raw_len)]
            reverse_list_label = [[] for i in range(raw_len)]
            labels_raw = reverse_labels[:, None].repeat(xy.shape[1], axis=1)

            for i in range(xy.shape[0]):
                for j in range(xy.shape[1]):
                    reverse_list_x[i + j].append(xy[i, j, 0])
                    reverse_list_y[i + j].append(xy[i, j, 1])
                    reverse_list_z[i + j].append(xy[i, j, 2])
                    # reverse_list_label[i + j].append(labels_raw[i, j])
                    reverse_list_label[i + j].extend(
                        np.ones(norm_label_weight[j], dtype='int8') * labels_raw[i, j])

            ##########cal_mean and error#######
            mean_x = np.array([np.mean(i) for i in reverse_list_x])
            mean_y = np.array([np.mean(i) for i in reverse_list_y])
            mean_z = np.array([np.mean(i) for i in reverse_list_z])

            mean_label = np.array([stats.mode(i)[0][0] for i in reverse_list_label])
            
            OFFSET = int(xy.shape[1] / 2)

            # return np.vstack([mean_x[OFFSET:-OFFSET], mean_y[OFFSET:-OFFSET],mean_z[OFFSET:-OFFSET]]).T, np.array(mean_label[OFFSET:-OFFSET])
            return np.vstack([mean_x, mean_y,mean_z]).T, np.array(mean_label)

        else:
            raw_len = xy.shape[1]-1+xy.shape[0]
            reverse_list_x = [[] for i in range(raw_len)]
            reverse_list_y = [[] for i in range(raw_len)]
            reverse_list_label = [[] for i in range(raw_len)]
            labels_raw = reverse_labels[:,None].repeat(xy.shape[1],axis=1)

            for i in range(xy.shape[0]):
                for j in range(xy.shape[1]):
                    reverse_list_x[i+j].append(xy[i,j,0])
                    reverse_list_y[i+j].append(xy[i,j,1])
                    # reverse_list_label[i+j].append(labels_raw[i,j])
                    reverse_list_label[i + j].extend(
                        np.ones(norm_label_weight[j], dtype='int8') * labels_raw[i, j])

            ##########cal_mean and error#######
            mean_x = np.array([np.mean(i) for i in reverse_list_x])
            # std_x  =np.array([np.std(i) for i in reverse_list_x])
            mean_y = np.array([np.mean(i) for i in reverse_list_y])
            # std_y = np.array([np.std(i) for i in reverse_list_y])

            mean_label = np.array([stats.mode(i)[0][0] for i in reverse_list_label])
            # plt.plot(mean_label)
            # plt.show()


            OFFSET = int(xy.shape[1] / 2)
            # std_t = (np.sqrt(std_x**2 +  std_y **2)/2).T
            # std_t = std_t[OFFSET:-OFFSET]
            # plt.plot(std_t)
            # plt.xlim([40000,56000])
            # plt.ylim([0,0.01])
            # plt.show()
            #52800
            # lo = 51000
            # hi = 54000
            # window_size = 12
            # a = 0.25
            # mean_x_plot = mean_x[lo:hi]
            # mean_x_plot_filter = self._get_slice_window(mean_x_plot,window_size,a)
            # mean_y_plot = mean_y[lo:hi]
            # mean_y_plot_filter = self._get_slice_window(mean_y_plot, window_size,a)
            # # std_x_plot = std_x[lo:hi]
            # # std_y_plot = std_y[lo:hi]
            # pos = self.pos[lo:hi,:]
            # labels_plot = mean_label[lo:hi]
            #
            # t = [i for i in range(len(mean_x_plot))]
            # # plt.plot(mean_x)
            #
            #
            # # plt.errorbar(t,mean_x_plot,yerr=std_x_plot)
            # plt.subplot(211)
            # plt.plot(pos[:,0])
            # plt.plot(mean_x_plot)
            #
            #
            # # plt.plot(mean_x_plot_filter[1:]-mean_x_plot_filter[:-1])
            # # plt.plot(mean_x_plot_filter)
            # # plt.show()
            # plt.subplot(212)
            # plt.plot(labels_plot)
            #
            # plt.show()
            #
            # # plt.plot(mean_x,std_x)
            # # plt.errorbar(t,mean_y_plot,yerr=std_y_plot)
            # plt.subplot(211)
            # plt.plot(pos[:, 1])
            # plt.plot(mean_y_plot)
            # # plt.plot(std_y_plot)
            # # plt.plot(mean_y_plot_filter)
            # # plt.plot(mean_y_plot_filter[1:]-mean_y_plot_filter[:-1])
            # # plt.show()
            # plt.subplot(212)
            # plt.plot(labels_plot)
            # plt.show()




            # t_save = np.array([i for i in range(len(mean_x))])
            # id = np.zeros(len(mean_x))
            # np.savetxt('./result/15min_2_smooth.csv', np.vstack([id,t_save,mean_x,mean_y]).T, fmt='%7f',delimiter=',')
            # np.savetxt('./result/15min_2_smooth_label.csv', self.kmeans.labels_, fmt='%d', delimiter=',')


            # return np.vstack([mean_x[OFFSET:-OFFSET],mean_y[OFFSET:-OFFSET]]).T, np.array(mean_label[OFFSET:-OFFSET])
            return np.vstack([mean_x,mean_y]).T, np.array(mean_label)

    def _get_slice_window(self,val,window=8,a=0.25):
        # num * 2
        X = np.ones(val.shape[0])*val[0]
        for hi in range(val.shape[0]):
            lo = max(hi - window,0)
            X[hi] = a*np.mean(val[lo:max(hi,1)]) + (1-a)*X[max(lo-1,0)] #one order filter

        return X








