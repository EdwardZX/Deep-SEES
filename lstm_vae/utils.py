import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from itertools import compress
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def data_processing(df,num_features,seq_len):
    #(t,x,y,feature,...)
    t = df.iloc[:, 0].values
    pos = df.iloc[:, 1:num_features+1].values
    data = df.iloc[:, num_features+1:].values.reshape(-1, num_features, seq_len).transpose(0, 2, 1)
    return data,(t,pos)


def load_data(path,filename_set):
    df = pd.DataFrame()
    idx = [0]
    for filename in filename_set:
        filename = path + filename + '.csv'
        # df = df.append(pd.read_csv(filename, header=None))
        df = pd.concat([df, pd.read_csv(filename, header=None)])
        idx.append(df.shape[0])

    df_msd = pd.DataFrame()
    for filename in filename_set:
        filename = path + filename + '_is_brownian' + '.csv'
        # df_msd = df_msd.append(pd.read_csv(filename, header=None))
        df_msd = pd.concat([df_msd, pd.read_csv(filename, header=None)])

    return df, df_msd, idx

def get_colors(centers):
    # color_set = np.array(sns.color_palette('rainbow'))
    num = len(centers)
    # cmap = plt.cm.get_cmap('rainbow',num)
    # steps = 1./num
    # color_set =np.array( [cmap(i) for i in np.arange(0,1,steps)])
    # color_set = cm.jet(np.linspace(0, 1, num))
    color_set = cm.rainbow(np.linspace(0, 1, num))

    # color_set[0] = color_set[-1]
    # color_set[-2]  = color_set[-1]
    #color_set = ['#FF0000', '#FFA500', '#800080', '#008000', '#0000FF','#FF1493','#8B0000','#483D8B']
    cluster_norm = np.linalg.norm(centers, axis=1, keepdims=True).reshape(-1)
    cluster_index = cluster_norm.argsort()
    return color_set,np.arange(len(centers))#cluster_index#np.arange(len(centers))

def plot_graph_every_time(df,centers,path_filename):
    time_list = sorted(df.iloc[:, 0].unique())
    color_set,cluster_index = get_colors(centers)
    #df = pd.concat([df, pd.DataFrame(alpha_set)], axis=1)
    cnt = 0
    step_size = 15
    trj_len = 30

    for v in range(0,len(time_list),step_size):
        cnt +=1
        if cnt < trj_len/step_size:
            for w in range(0, v, 1):
                df_list = list(df.index[df.iloc[:, 0] == time_list[w]].values)# get the index
                df_step = df.iloc[df_list,:]
                for i in range(df_step.shape[0]):
                    h = plt.scatter(df_step.iloc[i,1:2],df_step.iloc[i,2:3],c = color_set[int(cluster_index[int(df_step.iloc[i,-1])]),:]
                                ,s=20)

        else:
            for w in range(max(v-trj_len,0),v,1):
                df_list = list(df.index[df.iloc[:, 0] == time_list[w]].values)  # get the index
                df_step = df.iloc[df_list, :]
                for i in range(df_step.shape[0]):
                        h = plt.scatter(df_step.iloc[i, 1:2], df_step.iloc[i, 2:3],
                            c=color_set[int(cluster_index[int(df_step.iloc[i, -1])]),:]
                            , s=20)
        plt.xlim([min(df.iloc[:,1:2].values),max(df.iloc[:,1:2].values)])
        plt.ylim([min(df.iloc[:,2:3].values),max(df.iloc[:,2:3].values)])
        plt.axis('equal')
        plt.savefig(path_filename+str(time_list[v])+'.png')
        plt.show(block=False)
        #eplt.pause(0.05)
        plt.close()
        #for i in range(len(new_labels)):
        print(time_list[v])

def extents(f):
  delta = f[1] - f[0]
  return [f[0] - delta/2, f[-1] + delta/2]

def plot_single_trajectory(xy,t,labels,centers,lo=0,steps_set=-1,sz=0.005, is_equal = True):
    ##add OFFSET to evaluate the total sequence information
    # OFFSET = int(30/2)
    # xy_raw = xy[:-OFFSET,:]
    # labels_raw = labels[OFFSET:,]
    # t_raw = t[:-OFFSET]

    ##2d
    xy_raw = xy[:,:2]
    labels_raw = labels
    t_raw = t
    # print(t.shape[0], xy_raw.shape[0])

    ####### mask ######
    steps = len(xy_raw)-lo
    if steps_set > 0 and steps_set+lo <len(xy_raw):
        steps = steps_set

    # lo = 34431
    # steps = 38930 - 34431
    # lo = 45310
    # steps = 2500
    # lo = 51500
    # steps = 2500

    xy = xy_raw[lo:lo+steps, ]
    labels = labels_raw[lo:lo+steps, ]
    t = t_raw[lo:lo+steps]

    delta_t = np.hstack([1, t[1:] - t[0:-1]])
    filter_t = (delta_t != 1)
    delta_t = list(compress(range(len(filter_t)), filter_t))
    # print(delta_t)
    delta_t = np.array(delta_t)
    delta_t = delta_t[delta_t < len(xy_raw)]


    # delta_t = delta_t[lo:lo+steps]
    ####plot_line####
    xy_line = np.hstack([xy[:-1,:],xy[1:,:]])
    # print(delta_t)
    xy_line = np.vstack([np.hstack([xy[0,:],xy[0,:]]),xy_line])
    if len(delta_t):
        # print(xy_line[delta_t,0:2])
        xy_line[delta_t,0:2] =  xy_line[delta_t,2:4]
    X = np.vstack([xy_line[:,0],xy_line[:,2]]).T
    Y = np.vstack([xy_line[:,1],xy_line[:,3]]).T


    delta_t = np.hstack([0, delta_t, len(xy)])
    # X=np.array([[0,1],[2,3]])
    # Y=np.array([[0,2],[3,4]])
    # plt.plot(X.T,Y.T)
    # plt.show()

    sz = sz * 6e5/len(labels) #0.02
    color_set, cluster_index = get_colors(centers)

    lb_centers = set(labels)
    # if is_filtering:
    #     for j in range(len(delta_t)-1):
    #         plt.plot(xy[int(delta_t[j]):int(delta_t[j + 1]), 0], xy[int(delta_t[j]):int(delta_t[j + 1]), 1]
    #                  , c='#D3D3D3', zorder=10)
    #     return
    for i in lb_centers:
        if i == -1:
            continue
        for j in range(len(delta_t)-1):
            plt.plot(xy[int(delta_t[j]):int(delta_t[j + 1]), 0], xy[int(delta_t[j]):int(delta_t[j + 1]), 1], c='#D3D3D3', zorder=10)

        # plt.plot(xy[:, 0], xy[:, 1], c='#D3D3D3', zorder=10)
        if len(xy) < 5000:
            x = X[labels == i, :].T
            y = Y[labels == i, :].T
            plt.plot(x, y, c=color_set[int(cluster_index[i]), :], zorder=50)
        else:
            plt.scatter(xy[labels==i,0],xy[labels==i,1],s=sz,c=color_set[int(cluster_index[i]),:],zorder=50)
        
        if is_equal:
            plt.axis('equal')

        #for j in range(len(delta_t)-1):
        #    plt.plot(xy[delta_t[j]:delta_t[j+1], 0], xy[delta_t[j]:delta_t[j+1], 1],c ='#D3D3D3' ,zorder=10)
        plt.show()

    out_mat = np.tile(labels_raw[lo:min(lo + steps, labels_raw.shape[0])], (int(steps / 10), 1))
    masked_array = np.ma.masked_where(out_mat == -1, out_mat)
    cmap = plt.cm.get_cmap('rainbow',len(centers))
    cmap.set_bad(color='#D3D3D3')
    plt.imshow(masked_array, cmap=cmap,vmin=0, vmax=len(centers) - 1)
    plt.colorbar()
    plt.show()

    #plt.plot(xy[:, 0], xy[:, 1], c='#D3D3D3', zorder=10)
    for j in range(len(delta_t) - 1):
        plt.plot(xy[int(delta_t[j]):int(delta_t[j + 1]), 0], xy[int(delta_t[j]):int(delta_t[j + 1]), 1], c='#D3D3D3',
                 zorder=10)
    # plt.show()
    for i in lb_centers:
        if i == -1:
            continue
        if len(xy) < 5000:
            x = X[labels == i, :].T
            y = Y[labels == i, :].T
            plt.plot(x, y, c=color_set[int(cluster_index[i]), :], zorder=50)
        else:
            plt.scatter(xy[labels == i, 0], xy[labels == i, 1], s=sz, c=color_set[int(cluster_index[i]), :], zorder=50)
        # plt.scatter(xy[labels==i,0],xy[labels==i,1],s=sz,c=color_set[int(cluster_index[i]),:],zorder=50)
    # plt.show()
    # #segment:
    # lo_cross = 56000
    # cross_step = 3000
    # out_mat = np.tile(labels[lo_cross:lo_cross+cross_step], (int(cross_step/ 10), 1))
    # plt.imshow(out_mat, cmap='rainbow')
    # # plt.imshow(out_mat, cmap='rainbow')
    # plt.colorbar()
    # plt.show()
    #
    #other






    print('segement plot finished')


def plot_single_trajectory_3d(xy, t, labels, centers, lo=0, steps_set=3000):

    xy_raw = xy
    labels_raw = labels
    t_raw = t

    ####### mask ######
    steps = 3000
    if steps_set > 0 and steps_set + lo < len(xy_raw):
        steps = steps_set

    xy = xy_raw[lo:lo + steps, ]
    labels = labels_raw[lo:lo + steps, ]
    t = t_raw[lo:lo + steps]

    delta_t = np.hstack([1, t[1:] - t[0:-1]])
    filter_t = (delta_t != 1)
    delta_t = list(compress(range(len(filter_t)), filter_t))
    delta_t = np.array(delta_t)
    delta_t = delta_t[delta_t < len(xy_raw)]
    delta_t = np.hstack([0, delta_t, len(xy)])

    sz = 0.05 * 6e5 / len(labels)  # 0.02
    color_set, cluster_index = get_colors(centers)

    lb_centers = set(labels)
    # if is_filtering:
    #     for j in range(len(delta_t)-1):
    #         plt.plot(xy[int(delta_t[j]):int(delta_t[j + 1]), 0], xy[int(delta_t[j]):int(delta_t[j + 1]), 1]
    #                  , c='#D3D3D3', zorder=10)
    #     return
    out_mat = np.tile(labels_raw[lo:min(lo + steps, labels_raw.shape[0])], (int(steps / 10), 1))
    # out_mat = np.tile(labels_raw[lo:min(lo + steps, labels_raw.shape[0])], (int(steps / 10), 1))
    plt.imshow(out_mat, cmap='rainbow')
    # plt.xlim([lo,min(lo + steps, labels_raw.shape[0])])
    plt.colorbar()
    plt.show()

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')


    for i in lb_centers:
        if i == -1:
            continue
        for j in range(len(delta_t) - 1):
            ax.plot(xy[int(delta_t[j]):int(delta_t[j + 1]), 0], xy[int(delta_t[j]):int(delta_t[j + 1]), 1],
                     xy[int(delta_t[j]):int(delta_t[j + 1]), 2],
                     '-',
                     alpha=0.5,
                     c='#D3D3D3',
                     zorder=0,
                     linewidth=0.5)


        ax.scatter(xy[labels == i, 0], xy[labels == i, 1],xy[labels == i, 2],
                   s=sz, c=color_set[int(cluster_index[i]), :], zorder=5)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_axis_off()

        # Hide grid lines
    ax.grid(False)

        # show the plot
    # plt.show()

def plot_single_time_series(xy,t,labels,centers,
                            lo=0,steps_set=-1,n_dim=3,
                            extend_para=0.2,sz=0.005):
    xy_raw = xy
    labels_raw = labels
    t_raw = t

    ####### mask ######
    steps = 3000
    if steps_set > 0 and steps_set + lo < len(xy_raw):
        steps = steps_set

    # lo = 34431
    # steps = 38930 - 34431
    # lo = 45310
    # steps = 2500
    # lo = 51500
    # steps = 2500

    xy = xy_raw[lo:lo + steps, ]
    labels = labels_raw[lo:lo + steps, ]
    t = t_raw[lo:lo + steps]

    sz = sz * 6e5 / len(labels)
    LW = 2 # 0.02
    color_set, cluster_index = get_colors(centers)

    lb_centers = set(labels)
    out_mat = np.tile(labels_raw[lo:min(lo + steps, labels_raw.shape[0])], (int(steps / 10), 1))

    masked_array = np.ma.masked_where(out_mat == -1, out_mat)
    cmap = plt.cm.get_cmap('rainbow',len(centers))
    # print(len(centers))
    cmap.set_bad(color='#D3D3D3')
    plt.imshow(masked_array, cmap=cmap, vmin=0, vmax=len(centers) - 1)
    plt.colorbar()
    plt.show()

    # plt.plot(xy[:, 0], xy[:, 1], c='#D3D3D3', zorder=10)
    n_dim = min(xy.shape[1],n_dim)
    if n_dim == 1:
        fig_size = (6, 2.5)
        fig, ax = plt.subplots(n_dim, figsize=fig_size)
        ax = [ax]
    elif n_dim == 2:
        fig_size = (6, 3.5)
        fig, ax = plt.subplots(n_dim, sharex=True, gridspec_kw={'hspace': 0}, figsize=fig_size)
    else: # n_dim=2
        fig_size = (6, 4.5)
        fig, ax = plt.subplots(n_dim, sharex=True, gridspec_kw={'hspace': 0}, figsize=fig_size)


    ##set y_value

    for i_dim in range(n_dim):
        Min = xy[:,i_dim].min()
        Min = (Min + extend_para * Min) if np.sign(Min) == -1 else (Min - Min * extend_para)
        Max = xy[:,i_dim].max()
        Max = Max + np.sign(Max) * extend_para * Max
        y_lim = [Min, Max]

        ax[i_dim].plot(
            t,
            xy[:, i_dim],
            '-',
            c='#D3D3D3',
            lw=LW,
         )

        # Lims
        ax[i_dim].set_xlim([lo, t[-1]])
        ax[i_dim].set_ylim(y_lim)
    # plt.show()
        for i in lb_centers:
            if i == -1:
                continue
            ax[i_dim].scatter(t[labels == i],xy[labels == i, i_dim], s=sz, c=color_set[int(cluster_index[i]), :], zorder=50)
        # plt.show()


def my_tsne(data,labels,centers):
    color_set, cluster_index = get_colors(centers)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(data)
    #X_tsne = pickle.load(open("./data/save_t_sne_kmeans.p", "rb"))
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # å½’ä¸€åŒ–
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        if labels[i] ==-1:
            continue
        plt.scatter(X_norm[i, 0], X_norm[i, 1], color=color_set[int(cluster_index[int(labels[i])]),:])
        # print('drawing {} points'.format(i))
        #plt.scatter(X_norm[i, 0], X_norm[i, 1], color=plt.cm.Set1(kmeans.labels_[i]))
    #plt.show()
    plt.xticks([])
    plt.yticks([])
    return tsne

def my_bar(labels,centers):
    color_set, cluster_index = get_colors(centers)
    label_index = set(labels)
    labels_num  = labels.shape[0]-len(labels[labels == -1])
    cnt_set = []

    for i in label_index:#range(len(cluster_index)):
        if i ==-1:
            continue
        cnt = len(labels[labels == i])/labels_num * 100
        plt.bar(cluster_index[i],cnt,color = color_set[int(cluster_index[i]),:])
        cnt_set.append(cnt)
    plt.ylim([0, 100])
    #plt.show()
    return cnt_set