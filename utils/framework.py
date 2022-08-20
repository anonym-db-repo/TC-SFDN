import argparse
import tensorflow as tf
import numpy as np
import os


def get_gpus():
    value = os.getenv('CUDA_VISIBLE_DEVICES', '4')
    value = value.split(',')
    return len(value)


def make_dirs(path: str):
    pos = path.rfind(os.sep)
    if pos < 0:
        raise Exception('Can not find the directory from the path', path)
    path = path[:pos]
    os.makedirs(path, exist_ok=True)


class Config:
    def __init__(self):
        self.lr = 0.001
        self.epoches = 100
        self.batch_size = 20
        self.save_path = './models/{name}/{name}'.format(name=self.get_name())
        self.sample_path = None
        self.logdir = './logs/{name}'.format(name=self.get_name())
        self.new_model = False
        self.gpus = get_gpus()
        self.stopped = False

    def get_name(self):
        raise Exception('get_name() is not re-defined.')

    def __repr__(self):
        attrs = self.get_attrs()
        result = ['%s=%s' % (key, attrs[key]) for key in attrs]
        return ', '.join(result)

    def get_attrs(self):
        result = {}
        for attr in dir(self):
            if attr.startswith('_'):
                continue
            value = getattr(self, attr)
            if value is None or type(value) in (int, float, bool, str):
                result[attr] = value
        return result

    def from_cmd(self):
        parser = argparse.ArgumentParser()
        attrs = self.get_attrs()
        for attr in attrs:
            value = attrs[attr]
            t = type(value)
            if t == bool:
                parser.add_argument('--' + attr, default=value, help='Default to %s' % value,
                                    action='store_%s' % ('false' if value else 'true'))
            else:
                parser.add_argument('--' + attr, type=t, default=value, help='Default to %s' % value)
        parser.add_argument('--call', type=str, default='train', help='Call method, by default call train()')
        a = parser.parse_args()
        for attr in attrs:
            setattr(self, attr, getattr(a, attr))

        self.call(a.call)

    def call(self, name):
        if name == 'train':
            self.train()
        elif name == 'test':
            self.test()
        else:
            print('Unknown method name ' + name, flush=True)

    def train(self):
        app = self.get_app()
        with app:
            app.train(self.get_ds_train(), self.get_ds_train())

    def get_ds_train(self):
        raise Exception('get_ds_train() is not defined.')

    def test(self):
        app = self.get_app()
        with app:
            app.test(self.get_ds_test())

    def get_ds_test(self):
        raise Exception('get_ds_test() is not defined.')

    def get_tensors(self):
        return Tensors(self)

    def get_sub_tensors(self, gpu_index):
        """
        Get the sub tensors for the specified gpu.
        :param gpu_index: the index (based on zero) of the GPU
        :return: the sub tensors which has the property 'inputs'
        """
        raise Exception('The get_sub_tensors() is not defined.')

    def get_app(self):
        return App(self)

    @staticmethod
    def get_optimizer(lr):
        return tf.train.AdamOptimizer(lr)


class Tensors:
    """
    提供train_ops, summary, lr, sub_ts[i]: {inputs, losses, private_tensors}
    """
    def __init__(self, config: Config):
        self.config = config
        self.sub_ts = []
        # with tf.variable_scope(config.get_name(), reuse=tf.AUTO_REUSE):   # None, True, the 2nd method to reuse variables
        with tf.variable_scope(config.get_name()) as scope:
            for i in range(config.gpus):
                #with tf.device('/gpu:%d' % i):
                with tf.device('/gpu:%d' % 4):
                    self.sub_ts.append(config.get_sub_tensors(i))
                    scope.reuse_variables()    # The 1st method to reuse variables
                    # tf.get_variable_scope().reuse_variables()

        #with tf.device('/gpu:0'):
        with tf.device('/gpu:4'):
            with tf.variable_scope('%s_train' % config.get_name()):
                losses = [ts.losses for ts in self.sub_ts]    # [gpus, losses]
                self.losses = tf.reduce_mean(losses, axis=0)  # [losses]

                self.lr = tf.placeholder(tf.float32, name='lr')
                opt = config.get_optimizer(self.lr)

                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    grads = self.compute_grads(opt)
                    self.apply_grads(grads, opt)

            for i in range(len(losses[0])):
                tf.summary.scalar('loss_%d' % i, self.get_loss_for_summary(self.losses[i]))
            self.summary = tf.summary.merge_all()

    def get_loss_for_summary(self, loss):
        return loss

    def apply_grads(self, grads, opt):
        self.train_ops = [opt.apply_gradients(gs) for gs in grads]

    def compute_grads(self, opt):
        grads = []
        for gpu_id, ts in enumerate(self.sub_ts):
            #with tf.device('/gpu:%d' % gpu_id):
            with tf.device('/gpu:%d' % 4):
                grads.append([opt.compute_gradients(loss) for loss in ts.losses])
        return [self.get_grads_mean(grads, i) for i in range(len(grads[0]))]

    @staticmethod
    def get_grads_mean(grads, loss_idx):
        # grads: [gpus, losses]
        grads = [gs[loss_idx] for gs in grads]  # [gpus, vars, 2]
        gpus = len(grads)

        vars = [pair[1] for pair in grads[0]]
        result = []
        for i, var in enumerate(vars):
            g = grads[0][i][0]
            if isinstance(g, tf.IndexedSlices):
                values = [gs[i][0].values/gpus for gs in grads]  # [gpus, -1, 200]
                values = tf.concat(values, axis=0)  # [-1, 200]
                indices = [gs[i][0].indices for gs in grads]  # [gpus, -1]
                indices = tf.concat(indices, axis=0)  # [-1]
                result.append((tf.IndexedSlices(values, indices), var))
            else:
                result.append((tf.reduce_mean([gs[i][0] for gs in grads], axis=0), var))
        return result


class App:
    def __init__(self, config: Config):
        self.config = config
        graph = tf.Graph()
        with graph.as_default():
            self.ts = config.get_tensors()
            cfg = tf.ConfigProto()
            cfg.allow_soft_placement = True
            cfg.gpu_options.allow_growth = True
            cfg.gpu_options.per_process_gpu_memory_fraction = 0.95
            self.session = tf.Session(config=cfg, graph=graph)
            self.saver = tf.train.Saver(tf.global_variables())
            if config.new_model:
                self.session.run(tf.global_variables_initializer())
                print('Use a new empty models')
            else:
                try:
                    self.saver.restore(self.session, config.save_path)
                    print('Restore models from %s successfully!' % config.save_path)
                except:
                    print('Fail to restore models from %s, use a new empty models instead!!!!!!' % config.save_path)
                    self.session.run(tf.global_variables_initializer())

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def train(self, ds_train, ds_validation):
        self.before_train()
        cfg = self.config
        ts = self.ts
        writer = tf.summary.FileWriter(cfg.logdir, self.session.graph)

        epoch = 0
        step = 0
        while epoch < cfg.epoches and not cfg.stopped:
            self.before_epoch(epoch)
            batch = 0
            ds_train = cfg.get_ds_train()
            while batch < ds_train.num_examples // (cfg.batch_size * cfg.gpus) and not cfg.stopped:
                self.before_batch(epoch, batch)
                feed_dict = self.get_feed_dict(ds_train)
                if len(ts.train_ops) == 1:
                    _, summary, sum_loss, loss1, loss2, loss3 = \
                        self.session.run([ts.train_ops[0], ts.summary, ts.losses[0], ts.sub_ts[0].loss1, ts.sub_ts[0].loss2, ts.sub_ts[0].loss3], feed_dict)
                    print('epoch: %d, batch: %d, loss: %.6f, loss_node: %.6f, loss_frame: %.6f, loss_video: %.6f' % (epoch, batch, sum_loss, loss1, loss2, loss3))
                else:
                    for train_op in ts.train_ops:
                        self.session.run(train_op, feed_dict)
                    summary = self.session.run(ts.summary, feed_dict)
                writer.add_summary(summary, step)
                step += 1
                self.after_batch(epoch, batch)
                batch += 1
            print('Epoch:', epoch, flush=True)

            # TODO Temporary
            ds_test = self.config.get_ds_test()
            sub_ts = self.ts.sub_ts[0]
            data1, data2, label, causal_matrix, fault_node_label, next_frame_label, fault_frame_label = ds_test.next_batch(
                100)

            node_predict, frame_predict, video_predict, losses = \
                self.session.run([sub_ts.node_predict, sub_ts.frame_predict, sub_ts.video_predict, sub_ts.losses],
                                 {sub_ts.x1: data1, sub_ts.A: self.config.A, sub_ts.x2: data2,
                                  sub_ts.causal_matrix: causal_matrix,
                                  sub_ts.fault_node_labels: fault_node_label,
                                  sub_ts.next_frame_labels: next_frame_label,
                                  sub_ts.fault_frame_labels: fault_frame_label})
            node_count = fault_node_label.shape[0] * fault_node_label.shape[1]
            node_precise = (np.sum(np.argmax(node_predict, axis=-1) == fault_node_label)) / node_count
            next_frame_precise = (np.sum(np.argmax(frame_predict, axis=-1) == next_frame_label)) / \
                                 next_frame_label.shape[0]
            fault_frame_precise = (np.sum(np.argmax(video_predict, axis=-1) == fault_frame_label)) / \
                                  fault_frame_label.shape[0]
            print('node_precise: %.4f, next_frame_precise: %.4f, fault_frame_precise: %.4f, loss: %.4f,' %
                  (node_precise, next_frame_precise, fault_frame_precise, losses[0]))

            self.after_epoch(epoch)
            epoch += 1
        self.after_train()

    def before_train(self):
        print('Training is started!', flush=True)

    def before_epoch(self, epoch):
        if epoch > 0:
            self.saver.restore(self.session, self.config.save_path)

    def before_batch(self, epoch, batch):
        pass

    def after_train(self):
        print('Training is finished!', flush=True)

    def after_epoch(self, epoch):
        self.save()
        # ts = self.ts.sub_ts[0]
        # ds = self.config.get_ds_test()
        #
        # data1, data2, label, causal_matrix, fault_node_label, next_frame_label, fault_frame_label = ds.next_batch(100)
        #
        # node_predict, frame_predict, video_predict, losses = \
        #     self.session.run([ts.node_predict, ts.frame_predict, ts.video_predict, ts.losses],
        #                 {ts.x1: data1, ts.A: self.config.A, ts.x2: data2, ts.causal_matrix: causal_matrix,
        #                  ts.fault_node_labels: fault_node_label,
        #                  ts.next_frame_labels: next_frame_label, ts.fault_frame_labels: fault_frame_label})
        # node_count = fault_node_label.shape[0] * fault_node_label.shape[1]
        # node_precise = (np.sum(np.argmax(node_predict, axis=-1) == fault_node_label)) / node_count
        # next_frame_precise = (np.sum(np.argmax(frame_predict, axis=-1) == next_frame_label)) / next_frame_label.shape[0]
        # fault_frame_precise = (np.sum(np.argmax(video_predict, axis=-1) == fault_frame_label)) / \
        #                       fault_frame_label.shape[0]
        # print('node_precise: %.4f, next_frame_precise: %.4f, fault_frame_precise: %.4f, loss: %.4f,' %
        #       (node_precise, next_frame_precise, fault_frame_precise, losses[0]))

    def after_batch(self, epoch, batch):
        pass

    def save(self):
        self.saver.save(self.session, self.config.save_path)
        print('Save models into', self.config.save_path, flush=True)

    def test(self, ds_test):
        pass

    def get_feed_dict(self, ds):
        result = {self.ts.lr: self.config.lr}
        for i in range(self.config.gpus):
            values = ds.next_batch(self.config.batch_size)
            for tensor, value in zip(self.ts.sub_ts[i].inputs, values):
                result[tensor] = value
            result[self.ts.sub_ts[i].A] = self.config.A

            # TODO causal_matrix
            # result[self.ts.sub_ts[i].causal_matrix] = self.config.causal_matrix
        return result
