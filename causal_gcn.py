import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from utils.norm import InstanceNormalization

import utils.framework as myf
from feeder.feeder_t import Feeder
from utils.st_gcn import STGCN
from utils.graph import Graph
from utils.kappa_loss import WeightedKappaLoss


class MyConfig(myf.Config):
    def __init__(self):
        super(MyConfig, self).__init__()

        self._ds = None
        self.A = self.get_a()
        self.batch_size = 128
        self.lr = 0.0001
        self.new_model = False

        self.num_classes = 40
        self.filters = 16
        self.temporal_kernel_size = 5
        self.temperature = 0.2

        self.data_path = './data/Skeleton_Data/NTU-RGB-D/balance_data_6/train_data_10_all_02.npy'
        #self.data_path = './data/PKUMMD/balance_data_4/train_data_M_02_personal.npy'
        self.label_path = './data/Skeleton_Data/NTU-RGB-D/x_sub/train_label.pkl'
        self.causal_path = './data/Skeleton_Data/NTU-RGB-D/x_sub/train_data_causal_matrices_20.npy'
        self.global_causal_path = './data/Skeleton_Data/NTU-RGB-D/x_sub/train_data_global_causal_matrices.npy'

    @staticmethod
    def get_a():
        graph = Graph(layout='ntu-rgb+d', strategy='spatial')
        return graph.A.astype(np.float32)

    def get_name(self):
        return 'st-gcn_pkummd_personal'

    def get_sub_tensors(self, gpu_index):
        return CausualGCN(self)

    def get_ds_train(self):
        return self.ds

    def get_ds_test(self):
        # return self.ds
        test_data_path = './data/Skeleton_Data/NTU-RGB-D/type_data_10/insert_data_type21_same.npy'
        #test_data_path = './data/PKUMMD/balance_data_4/test_data_M_02_personal.npy'
        test_label_path = './data/Skeleton_Data/NTU-RGB-D/x_sub/val_label.pkl'
        causal_path = './data/Skeleton_Data/NTU-RGB-D/x_sub/val_data_causal_matrices.npy'
        global_causal_path = './data/Skeleton_Data/NTU-RGB-D/x_sub/val_data_global_causal_matrices.npy'
        return MyDS(self.batch_size, test_data_path, test_label_path, causal_path, global_causal_path)

    @property
    def ds(self):
        if self._ds is None:
            self._ds = MyDS(self.batch_size, self.data_path, self.label_path, self.causal_path, self.global_causal_path)
        return self._ds

    def get_app(self):
        return MyApp(self)


class CausualGCN:
    def __init__(self, cfg: MyConfig):
        # x: [N, C, T, V]   y: [N]
        self.config = cfg

    def build(self, A):
        x1 = layers.Input([50, 25, 3], name='x1')
        x2 = layers.Input([20, 25, 3], name='x2')
        # self.fault_node_labels = layers.Input([25], name='fault_node_labels')
        # self.next_frame_labels = layers.Input([1], name='next_frame_labels')
        # self.fault_frame_labels = layers.Input([1], name='fault_frame_labels')
        y = layers.Input([1], name='y')
        causal_matrix = layers.Input([1, 25, 25], name='causal_matrix')
        inputs = [x1, x2, y, causal_matrix]
        st_gcn_net = STGCN()

        x1 = self.data_bn(x1, 'data_x1')  # [-1, 50, 25, 3]
        x2 = self.data_bn(x2, 'data_x2')  # [-1, 20, 25, 3]
        filters = self.config.filters
        # importance_0 = tf.get_variable('importance', self.A.shape, tf.float32)
        # adjacency_matrix = self.A * importance_0
        adjacency_matrix = K.constant(A, dtype=K.floatx(), name='adj_matrix')  # layers.Lambda(lambda x: x[0])(self.A)
        kernel_size = (self.config.temporal_kernel_size, adjacency_matrix.shape[0])  # [5, 1]

        # GCN * 3
        for i in range(3):
            filters *= 2
            x1 = st_gcn_net.gcn(x1, adjacency_matrix, filters, kernel_size[1], name='x1_gcn_%d' % i)
            x2 = st_gcn_net.gcn(x2, adjacency_matrix, filters, kernel_size[1], name='x2_gcn_%d' % i)

        x1_raw = st_gcn_net.residual(x1, residual=True, in_filters=0, out_filters=filters, normalize=True)
        x2_raw = st_gcn_net.residual(x2, residual=True, in_filters=0, out_filters=filters, normalize=True)
        x1 += x1_raw
        x2 += x2_raw

        # causal_GCN * 3
        for i in range(3):
            x1 = st_gcn_net.causal_gcn(x1, causal_matrix, filters, causal_matrix.shape[1],
                                            name='x1_causal_gcn_%d' % i)
            x2 = st_gcn_net.causal_gcn(x2, causal_matrix, filters, causal_matrix.shape[1],
                                            name='x2_causal_gcn_%d' % i)

        x1_raw = st_gcn_net.residual(x1_raw, residual=True, in_filters=filters, out_filters=filters, normalize=True)
        x2_raw = st_gcn_net.residual(x2_raw, residual=True, in_filters=filters, out_filters=filters, normalize=True)
        x1 += x1_raw
        x2 += x2_raw

        x1_node_semantics = x1  # [-1, 50, 25, 512]
        x2_node_semantics = x2  # [-1, 20, 25, 512]

        # TCN * 3
        i = 0
        while x1_node_semantics.shape[1] > kernel_size[0]:
            i += 1
            x1_node_semantics = st_gcn_net.tcn(x1_node_semantics, x1_node_semantics.shape[-1], kernel_size[0],
                                                    2, 0, name='x1_tcn_%d' % i)

        x1_node_semantics = st_gcn_net.tcn(x1_node_semantics, x1_node_semantics.shape[-1], x1_node_semantics.shape[1],
                                           stride=1, padding='valid', name='x1_tcn_output')  # [-1, 1, 25,512]
        # x1_node_semantics = K.squeeze(x1_node_semantics, axis=1)  # [-1, 25, 512]

        # TCN * 2
        i = 0
        while x2_node_semantics.shape[1] > kernel_size[0]:
            i += 1
            x2_node_semantics = st_gcn_net.tcn(x2_node_semantics, x2_node_semantics.shape[-1], kernel_size[0],
                                               2, 0, name='x2_tcn_%d' % i)
        x2_node_semantics = st_gcn_net.tcn(x2_node_semantics, x2_node_semantics.shape[-1], x2_node_semantics.shape[1],
                                           stride=1, padding='valid', name='x2_tcn_output')  # [-1, 1, 25, 512]
        # x2_node_semantics = K.squeeze(x2_node_semantics, axis=1)  # [-1, 25, 512]

        l2_norm_layer = layers.Lambda(lambda x: K.l2_normalize(x, axis=-1))

        mid_act_fun = layers.LeakyReLU()      # 'tanh'
        output_act_fun = 'tanh'

        order = 3
        x1_recent = x1[:, -order:]  # st_gcn_net.tcn(x1[:, -order:], nb_joint_feat, order, stride=1, padding='valid')
        x1_recent = K.permute_dimensions(x1_recent, [0, 2, 1, 3])
        x1_recent = layers.Reshape([x1_recent.shape[1], -1])(x1_recent)
        # x1_recent = x1_node_semantics
        # x1_recent = K.squeeze(x1_recent, axis=1)

        # Joint-level forgery
        nb_joint_feat = 256
        x1_node_rep = layers.Dense(nb_joint_feat, activation=mid_act_fun, name='x1_joint_dense1')(x1_recent)
        x1_node_rep = K.tile(K.expand_dims(x1_node_rep, axis=1), [1, x2.shape[1], 1, 1])
        x2_node_rep = x2    # x2[:, 0]  # [-1, 25, 256]
        x2_node_rep = layers.Dense(nb_joint_feat, activation=mid_act_fun, name='x2_joint_dense1')(x2_node_rep)

        # x1_node_rep = l2_norm_layer(x1_node_rep)
        # x2_node_rep = l2_norm_layer(x2_node_rep)

        x1_x2_merge_interact = layers.concatenate([x1_node_rep, x2_node_rep])  # [-1, 25, 512]
        # x1_x2_merge_interact = InstanceNormalization()(x1_x2_merge_interact)
        x1_x2_merge_interact = layers.Dense(nb_joint_feat, activation=output_act_fun)(x1_x2_merge_interact)

        # x1_x2_merge_interact = layers.multiply([x1_node_rep, x2_node_rep])
        self.node_score = layers.Dense(3, activation='softmax', name='node_score')(x1_x2_merge_interact[:, 0])

        # Next-frame detection
        nb_frame_feat = 256

        # model I
        # x1_frame_semantics = layers.Flatten()(x1_node_semantics)    # [-1, 25*128]
        # x1_frame_semantics = layers.Dense(nb_frame_feat, activation=output_act_fun, name='x1_frame_dense1')(x1_frame_semantics)  # [-1, 1024]
        # x1_frame_semantics = l2_norm_layer(x1_frame_semantics)
        #
        # x2_frame_semantics = layers.Reshape([x2.shape[1], -1])(x2)    # [-1, 20, 25*128]
        # x2_frame_semantics = layers.Dense(nb_frame_feat, activation=output_act_fun, name='x2_frame_semantics')(x2_frame_semantics)  # [-1, 1024]
        # x2_frame_semantics = l2_norm_layer(x2_frame_semantics)
        #
        # # x1_frame_semantics = layers.RepeatVector(x2.shape[1])(x1_frame_semantics)
        # x1_x2_frame_interact = layers.multiply([x1_frame_semantics, x2_frame_semantics])
        # # x1_x2_frame_interact = layers.Dense(256, activation=output_act_fun)(x1_x2_frame_interact)
        # x1_x2_frame_interact = layers.Dense(1)(x1_x2_frame_interact)  # frame score
        # x1_x2_frame_interact = K.squeeze(x1_x2_frame_interact, axis=-1)
        # # x1_x2_frame_interact = K.batch_dot(x2_frame_semantics, x1_frame_semantics) / self.config.temperature

        # model II
        # x1_frame_semantics = layers.Dense(nb_frame_feat, activation=mid_act_fun, name='x1_frame_dense1')(x1_recent)
        # # x1_frame_semantics = l2_norm_layer(x1_frame_semantics)
        # x1_frame_semantics = K.tile(K.expand_dims(x1_frame_semantics, axis=1), [1, x2.shape[1], 1, 1])
        # x2_frame_semantics = layers.Dense(nb_frame_feat, activation=mid_act_fun, name='x2_frame_dense1')(x2)
        # # x2_frame_semantics = l2_norm_layer(x2_frame_semantics)
        #
        # x1_x2_frame_interact = layers.concatenate([x1_frame_semantics, x2_frame_semantics])  # [-1, 20, 25, 512]
        # x1_x2_frame_interact = layers.Dense(nb_frame_feat, activation=output_act_fun)(x1_x2_frame_interact)

        x1_x2_frame_interact = K.mean(x1_x2_merge_interact, axis=-2)
        # x1_x2_frame_interact = layers.Reshape([x1_x2_merge_interact.shape[1], -1])(x1_x2_merge_interact)  # [-1, 20, 25*256]
        # x1_x2_frame_interact = layers.Dense(nb_frame_feat*2, activation=output_act_fun)(x1_x2_frame_interact)
        x1_x2_frame_interact = layers.Dense(1)(x1_x2_frame_interact)   # frame score
        x1_x2_frame_interact = K.squeeze(x1_x2_frame_interact, axis=-1)

        self.frame_score = layers.Activation(activation='softmax', name='frame_score')(x1_x2_frame_interact)
        # self.frame_score = layers.Activation(activation='sigmoid', name='frame_score')(x1_x2_frame_interact)

        def nce_loss(y_true, y_pred):
            y_true = K.one_hot(K.cast(y_true, np.int32), K.shape(y_pred)[-1])
            return K.mean(keras.losses.binary_crossentropy(y_true, y_pred))

        # Action-level forgery
        nb_video_feat = 256
        # model I
        # x1_video_semantics = layers.Flatten()(x1_node_semantics)  # [-1, 25*128]
        # x1_video_semantics = layers.Dense(nb_video_feat, activation=output_act_fun, name='x1_video_dense1')(x1_video_semantics)  # [-1, 1024]
        # x1_video_semantics = l2_norm_layer(x1_video_semantics)
        #
        # x2_video_semantics = layers.Flatten()(x2_node_semantics)  # [-1, 25*128]
        # x2_video_semantics = layers.Dense(nb_video_feat, activation=output_act_fun, name='x2_video_dense1')(x2_video_semantics)  # [-1, 512]
        # x2_video_semantics = l2_norm_layer(x2_video_semantics)
        #
        # x1_x2_video_interact = layers.multiply([x1_video_semantics, x2_video_semantics])

        # model II
        normalize = True
        mid_act_fun = layers.LeakyReLU()  # 'tanh'
        output_act_fun = 'tanh'
        x1_video_semantics = layers.Dense(nb_video_feat, activation=mid_act_fun, name='x1_video_dense1')(x1_node_semantics)
        x2_video_semantics = layers.Dense(nb_video_feat, activation=mid_act_fun, name='x2_video_dense1')(x2_node_semantics)
        if normalize:
            x1_video_semantics = l2_norm_layer(x1_video_semantics)
            x2_video_semantics = l2_norm_layer(x2_video_semantics)

        x1_x2_video_interact = layers.concatenate([x1_video_semantics, x2_video_semantics])
        x1_x2_video_interact = K.squeeze(x1_x2_video_interact, axis=1)

        x1_x2_video_interact = layers.Dense(nb_video_feat, activation=output_act_fun)(x1_x2_video_interact)
        # x1_x2_video_interact = layers.Reshape([x1_x2_video_interact.shape[1], -1])(x1_x2_video_interact)  # [-1, 20, 25*256]
        # x1_x2_video_interact = layers.Flatten()(x1_x2_video_interact)
        x1_x2_video_interact = K.mean(x1_x2_video_interact, axis=-2)

        self.video_score = layers.Dense(3, activation='softmax', name='video_score')(x1_x2_video_interact)

        def sparse_kappa_loss(y_true, y_pred):
            num_class = K.shape(y_pred)[-1]
            y_true = K.one_hot(K.cast(y_true, np.int32), num_class)
            kappa_loss = WeightedKappaLoss(num_class)
            return kappa_loss(y_true, y_pred)

        self.model = keras.models.Model(inputs=inputs, outputs=[self.node_score, self.frame_score, self.video_score])

        self.model.compile(optimizer='Adam',
                           loss=[sparse_kappa_loss,
                                 keras.losses.SparseCategoricalCrossentropy(),
                                 sparse_kappa_loss],
                           loss_weights=[1, 1, 1],
                           metrics={'frame_score': keras.metrics.sparse_categorical_accuracy,
                                    'video_score': keras.metrics.sparse_categorical_accuracy})
        return self.model

    def data_bn(self, x, name):
        # input: [N, T, V, C]  output: [N, T, V, C]
        # t, v, c = x.shape[1:]
        # x = tf.transpose(x, [0, 2, 3, 1])  # [N, V, C, T]
        # x = tf.reshape(x, [-1, v * c, t])  # [N, V*C, T]
        # x = layers.BatchNormalization(axis=1, name='bn_%s' % name)(x)  # [N, V*C, T]
        # x = tf.reshape(x, [-1, v, c, t])  # [N, V, C, T]
        # x = tf.transpose(x, [0, 3, 1, 2])   # [N, T, V, C]
        x = InstanceNormalization()(x)
        return x


class MyApp(myf.App):
    def after_epoch(self, epoch):
        super(MyApp, self).after_epoch(epoch)
        p_node, p_next_frame, p_fault_frame, loss = predict()
        print('node_precise: %.4f, next_frame_precise: %.4f, fault_frame_precise: %.4f, loss: %.4f,' %
              (p_node, p_next_frame, p_fault_frame, loss))


class MyDS:
    def __init__(self, batch_size, data_path, label_path, causal_path, global_causal_path):
        self.start = 0  # np.random.randint(0, 20)
        self.batch_size = batch_size
        self.feeder = Feeder(data_path, label_path, causal_path, global_causal_path)
        self.num_examples = len(self.feeder)
        self.batches = int(np.ceil(self.num_examples / batch_size))

    def reset(self):
        self.start = 0

    def next_batch(self):
        start = self.start
        end = min(start + self.batch_size, self.num_examples)
        self.start = end
        return self.feeder[start:end]


def predict():
    test_data_path = './data/Skeleton_Data/NTU-RGB-D/x_sub/train_data.npy'
    test_label_path = './data/Skeleton_Data/NTU-RGB-D/x_sub/train_label.pkl'
    causal_path = './data/Skeleton_Data/NTU-RGB-D/x_sub/train_data_causal_matrices_20.npy'
    global_causal_path = './data/Skeleton_Data/NTU-RGB-D/x_sub/train_data_global_causal_matrices.npy'
    ds = MyDS(100, test_data_path, test_label_path, causal_path, global_causal_path)

    data1, data2, label, causal_matrix, fault_node_label, next_frame_label, fault_frame_label = ds.next_batch(100)

    config = MyConfig()

    app = myf.App(config)
    session = app.session
    ts = app.ts.sub_ts[0]
    # params = tf.trainable_variables()
    # print("Trainable variables:------------------------")
    # for idx, v in enumerate(params):
    #     print("  param {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
    #
    # W = session.run(params[26])
    # b = session.run(params[27])


    node_predict, frame_predict, video_predict, losses = \
        session.run([ts.node_predict, ts.frame_predict, ts.video_predict, ts.losses],
                    {ts.x1: data1, ts.A: config.A, ts.x2: data2, ts.causal_matrix: causal_matrix, ts.fault_node_labels: fault_node_label,
                     ts.next_frame_labels: next_frame_label, ts.fault_frame_labels: fault_frame_label})
    node_count = fault_node_label.shape[0] * fault_node_label.shape[1]
    node_precise = (np.sum(np.argmax(node_predict, axis=-1) == fault_node_label)) / node_count
    next_frame_precise = (np.sum(np.argmax(frame_predict, axis=-1) == next_frame_label)) / next_frame_label.shape[0]
    fault_frame_precise = (np.sum(np.argmax(video_predict, axis=-1) == fault_frame_label)) / fault_frame_label.shape[0]
    return node_precise, next_frame_precise, fault_frame_precise, losses[0]
    # precise = np.sum((predict == test_label).astype(np.int64)) / len(test_label)
    # return test_label, predict, precise, np.around(loss, 4)


if __name__ == '__main__':
    my_config = MyConfig()
    # my_config.from_cmd()
    model = CausualGCN(my_config)
    model.build(np.zeros([3,25,25]))
    # p_node, p_next_frame, p_fault_frame, loss = predict()
    # print('node_precise: %.4f, next_frame_precise: %.4f, fault_frame_precise: %.4f, loss: %.4f,' %
    #       (p_node, p_next_frame, p_fault_frame, loss))
