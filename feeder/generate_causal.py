import numpy as np

import tools
import feeder


if __name__ == '__main__':

    # data_path = '../data/Skeleton_Data/NTU-RGB-D/x_sub/train_data.npy'
    label_path = '../data/Skeleton_Data/NTU-RGB-D/x_sub/train_label.pkl'
    # causal_path = '../data/Skeleton_Data/NTU-RGB-D/x_sub/train_data_causal_matrices_10.npy'
    # global_path = '../data/Skeleton_Data/NTU-RGB-D/x_sub/train_data_causal_matrices_10.npy'
    #
    # feeder = feeder.Feeder(data_path, label_path, causal_path, global_path)
    # print(len(feeder))  # 7194
    # # datas: [7194, 3, 300, 25, 2]
    # data_feeder = feeder[:]
    # all_data = data_feeder[0][..., 0]  # [N, C, T, V]
    #
    # n, c, t, v = all_data.shape
    #
    # causal_matrices = []
    # for i in range(n):
    #     data = all_data[i]  # [C, T, V]
    #     index = np.sum(data, axis=(0, 2)) != 0
    #     # if len(index) > 3:
    #     #     index = index[2]
    #     # else:
    #     #     index = index[-1]
    #     if np.sum(index) > 70:
    #         frame_data = data[:, index, :] # [C, T, V]
    #         frame_data = np.transpose(frame_data, [2, 0, 1])  # [V, C, T]
    #         causal_matrix = tools.pTE(frame_data, model_order=4, to_norm=False)
    #     else:
    #         causal_matrix = np.zeros([v, v])
    #     causal_matrices.append(causal_matrix)
    #     print(i)

    # np.save('../data/Skeleton_Data/NTU-RGB-D/x_sub/train_data_causal_matrices_20.npy', np.array(causal_matrices))

    # compute category-level causal matrices
    matrices = np.load('../data/Skeleton_Data/NTU-RGB-D/x_sub/train_data_causal_matrices_20.npy')  # [-1. 25, 25]
    _, labels = np.load(label_path, allow_pickle=True)
    labels = np.array(labels)

    global_matrices = []

    for i in range(40):
        label_index = np.where(labels == i)[0]
        type_matrix = matrices[label_index]  # [-1, 25, 25]
        global_matrix = np.mean(type_matrix, axis=0)

        global_matrices.append(global_matrix)

    np.save('../data/Skeleton_Data/NTU-RGB-D/x_sub/train_data_global_causal_matrices.npy', global_matrices)

