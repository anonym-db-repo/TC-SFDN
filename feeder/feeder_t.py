import numpy as np


class Feeder:
    def __init__(self, data_file, label_file, causal_file, global_causal_file, debug=False, mmap=True):
        self.data_path = data_file
        self.label_path = label_file
        self.causal_path = causal_file
        self.global_causal_path = global_causal_file
        self.debug = debug

        self.types_indexes = []
        self.fault_node_labels = []
        self.next_frame_labels = []
        self.fault_frame_labels = []

        self.data = None  # global_data
        self.frame_data = None
        self.spdata = []
        self.label = None
        self.sample_labels = []
        self.causal_matrices = None
        self.type_causal_matrices = None
        self.personal_causal_matrices = []

        # random.seed(1234)
        self.load_data(mmap)

    def shuffle(self, to_shuffle=True):
        idx = np.arange(self.train_data.shape[0])
        if to_shuffle:
            np.random.shuffle(idx)

        self.spdata = np.array(list(self.train_data[:, 0]))[idx]  # [data for data, _, _, _, _, _, _ in t_data]
        self.sample_labels = np.array(list(self.train_data[:, 1]))[idx]  # [label for _, label, _, _, _, _, _ in t_data]
        self.personal_causal_matrices = np.array(list(self.train_data[:, 2]))[
            idx]  # [causal_matrix for _, _, causal_matrix, _, _, _, _ in t_data]
        self.fault_node_labels = np.array(list(self.train_data[:, 3]))[
            idx]  # [node_label for _, _, _, node_label, _, _, _ in t_data]
        self.next_frame_labels = np.array(list(self.train_data[:, 4]))[
            idx]  # [next_label for _, _, _, _, next_label, _, _ in t_data]
        self.fault_frame_labels = np.array(list(self.train_data[:, 5]))[
            idx]  # [frame_label for _, _, _, _, _, frame_label, _ in t_data]
        self.original_data = np.array(list(self.train_data[:, 6]))[
            idx]  # [original for _, _, _, _, _, _, original in t_data]

    def load_data(self, mmap):
        self.train_data = np.load(self.data_path, allow_pickle=True)
        self.shuffle(False)

    def __len__(self):
        return len(self.spdata)

    def __getitem__(self, index):
        # get data
        # original_data = np.array(self.data[index])
        data = self.spdata[index]
        label = self.sample_labels[index] # np.array(, dtype=np.float32)[index]
        causal_matrix = self.personal_causal_matrices[index]
        fault_node_label = self.fault_node_labels[index]
        next_frame_label = self.next_frame_labels[index]
        fault_frame_label = self.fault_frame_labels[index]

        return [data[:, :50], data[:, 50:], label, causal_matrix[:, np.newaxis]], [fault_node_label, next_frame_label, fault_frame_label]

#
# if __name__ == '__main__':
#     data_path = '../data/Skeleton_Data/NTU-RGB-D/x_sub/train_data.npy'
#     label_path = '../data/Skeleton_Data/NTU-RGB-D/x_sub/train_label.pkl'
#     causal_path = '../data/Skeleton_Data/NTU-RGB-D/x_sub/train_data_causal_matrices_20.npy'
#     global_causal_path = '../data/Skeleton_Data/NTU-RGB-D/x_sub/train_data_global_causal_matrices.npy'
#
#     feeder = Feeder(data_path, label_path, causal_path, global_causal_path)
#     # print(len(feeder))  # 7194
#     # datas: [7194, 3, 300, 25, 2]
#     data, label, frame_index, node_classify  = feeder[:]
#     print(data[0][:, 0, :, 0])
#     print(data[0].shape)
