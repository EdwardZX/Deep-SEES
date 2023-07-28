from scipy.spatial.distance import euclidean
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import pickle
import os
from sklearn import manifold
import seaborn as sns
# import imagesc

def data_processing(df):
    id = df.iloc[:, 0].values
    data = df.iloc[:, 3:].values
    label = df.iloc[:, -1].values
    return id, data, label

def get_colors(centers):
    # color_set = np.array(sns.color_palette('rainbow'))
    num = len(centers)
    cmap = plt.cm.get_cmap('rainbow',num)
    steps = 1./num
    color_set =np.array( [cmap(i) for i in np.arange(0,1,steps)])
    #color_set = ['#FF0000', '#FFA500', '#800080', '#008000', '#0000FF','#FF1493','#8B0000','#483D8B']
    cluster_norm = np.linalg.norm(centers, axis=1, keepdims=True).reshape(-1)
    cluster_index = cluster_norm.argsort()
    return color_set,np.arange(len(centers))

def plot_graph_every_time(df,centers,path_filename):
    id, data, labels = data_processing(df)
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

def plot_single_trajectory(df,labels,centers):
    ##add OFFSET to evaluate the total sequence information
    OFFSET = int(30/2)
    xy = df.iloc[:,1:3].values

    xy_raw = xy[:-OFFSET,:]
    labels_raw = labels[OFFSET:,]
    ####### mask ######
    lo = 0
    steps = len(xy_raw)

    # lo = 34431
    # steps = 38930 - 34431
    # lo = 45310
    # steps = 2500
    lo = 51500
    steps = 2500

    xy = xy_raw[lo:lo+steps, ]
    labels = labels_raw[lo:lo+steps, ]

    ####plot_line####
    xy_line = np.hstack([xy[:-1,:],xy[1:,:]])
    xy_line = np.vstack([np.hstack([xy[0,:],xy[0,:]]),xy_line])
    X = np.vstack([xy_line[:,0],xy_line[:,2]]).T
    Y = np.vstack([xy_line[:,1],xy_line[:,3]]).T

    # X=np.array([[0,1],[2,3]])
    # Y=np.array([[0,2],[3,4]])
    # plt.plot(X.T,Y.T)
    # plt.show()

    sz = 0.02 * 6e5/len(labels)
    color_set, cluster_index = get_colors(centers)

    lb_centers = set(labels)

    for i in lb_centers:
        if i == -1:
            continue
        plt.plot(xy[:, 0], xy[:, 1], c='#D3D3D3', zorder=10)
        if len(xy) < 5000:
            x = X[labels == i, :].T
            y = Y[labels == i, :].T
            plt.plot(x, y, c=color_set[int(cluster_index[i]), :], zorder=50)
        else:
            plt.scatter(xy[labels==i,0],xy[labels==i,1],s=sz,c=color_set[int(cluster_index[i]),:],zorder=50)

        plt.plot(xy[:, 0], xy[:, 1],c ='#D3D3D3' ,zorder=10)
        plt.show()

    out_mat = np.tile(labels_raw[lo:min(lo + steps, labels_raw.shape[0])], (int(steps / 10), 1))
    plt.imshow(out_mat, cmap='rainbow')
    plt.colorbar()
    plt.show()


    plt.plot(xy[:, 0], xy[:, 1], c='#D3D3D3', zorder=10)
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


def my_tsne(data,labels,centers):
    color_set, cluster_index = get_colors(centers)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(data)
    #X_tsne = pickle.load(open("./data/save_t_sne_kmeans.p", "rb"))
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.scatter(X_norm[i, 0], X_norm[i, 1], color=color_set[int(cluster_index[int(labels[i])]),:])
        print('drawing {} points'.format(i))
        #plt.scatter(X_norm[i, 0], X_norm[i, 1], color=plt.cm.Set1(kmeans.labels_[i]))
    #plt.show()
    return tsne

def my_bar(labels,centers):
    color_set, cluster_index = get_colors(centers)
    label_index = set(labels)
    labels_num  = labels.shape[0]-len(labels[labels == -1])
    for i in range(len(cluster_index)):
        cnt = len(labels[labels == i])/labels_num * 100
        plt.bar(cluster_index[i],cnt,color = color_set[int(cluster_index[i]),:])

    plt.ylim([0, 100])
    #plt.show()

def my_scalar(raw,n_feature):
    col = raw.shape[1]
    # multi_feature
    num_cols = int(col / n_feature)
    for i in range(n_feature):
        if i == 0:
            a = raw[:, i * num_cols:(i + 1) * num_cols]
            a_min = np.min(raw[:, i * num_cols:(i + 1) * num_cols])
            a_max = np.max(raw[:, i * num_cols:(i + 1) * num_cols])
            scalar_data = (a-a_min)/(a_max-a_min)
            continue
        else:
            a = raw[:, i * num_cols:(i + 1) * num_cols]
            a_min = np.min(raw[:, i * num_cols:(i + 1) * num_cols])
            a_max = np.max(raw[:, i * num_cols:(i + 1) * num_cols])
            a = (a - a_min) / (a_max - a_min)
            scalar_data = np.hstack((scalar_data, a))
    return scalar_data


if __name__ == '__main__':
    #filename_test = 'data/graph_data_set.csv'
    filename = 'data/graph_data_set.csv'
    df = pd.read_csv(filename, header=None)
    id,data,label = data_processing(df)
    if os.path.isfile('./data/save_kmeans.p'):
        kmeans = pickle.load(open("./data/save_kmeans.p", "rb"))
    else:
        kmeans = KMeans(n_clusters=5, random_state=0).fit(data)
        pickle.dump(kmeans, open("./data/save_kmeans.p", "wb"))

    #df.drop(columns=[-1], inplace=True,axis=1)
    df = pd.concat([df.iloc[:,:-1], pd.DataFrame(kmeans.labels_)], axis=1)
    #plot_graph_every_time(df, kmeans.cluster_centers_)
    tsne = my_tsne(data,kmeans.labels_,kmeans.cluster_centers_)
    pickle.dump(tsne, open("./data/save_t_sne_kmeans.p", "wb"))
