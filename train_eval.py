from causal_gcn import MyConfig, CausualGCN
import tensorflow as tf
from metric.ranking import precisionAtK, recallAtK, auc
from sklearn.metrics import accuracy_score
import numpy as np
from pathlib import Path
import scipy.io as io

# tf.config.experimental_run_functions_eagerly(True)
tf.compat.v1.disable_v2_behavior()

cfg = MyConfig()
# cfg.from_cmd()
causalgraph = CausualGCN(cfg)
causalgraph.build(cfg.A)
save_path = './models/TCGCN'

ds_train = cfg.get_ds_train()
ds_test = cfg.get_ds_test()

print_test_loss = False
test_per_epoch = 1
epoch = 0
step = 0
while epoch < cfg.epoches and not cfg.stopped:
    batch = 0
    print('Star epoch %d......' % epoch)
    ds_train.feeder.shuffle()
    ds_train.reset()
    while batch < ds_train.batches and not cfg.stopped:
        ds_batch_x, ds_batch_y = ds_train.next_batch()
        sum_loss, joint_loss, frame_loss, video_loss, frame_acc, video_acc =\
            causalgraph.model.train_on_batch(x=ds_batch_x, y=ds_batch_y)
        print('Batch %d, Loss %g, Joint Loss %g, Frame Loss %g, Video Loss %g' % (batch, sum_loss, joint_loss, frame_loss, video_loss))
        print('Batch %d, Frame Acc %g, Video Acc %g' % (batch, frame_acc, video_acc))
        causalgraph.model.save_weights(save_path)
        step += 1
        batch += 1
    print('Finish epoch ', epoch)
    if epoch % test_per_epoch == 0:
        print('Start evaluation...')
        print('==============================================================')
        # causalgraph.model.save_weights(cfg.get_name())
        # causalgraph.model.load_weights(cfg.get_name())
        joint_rec_list = []
        joint_prc_list = []
        frame_auc_list = []
        frame_mrr_list = []
        frame_ar_list = []
        frame_arr_list = []
        nsf_acc_list = []
        video_acc_list = []
        video_accd_list = []
        video_label_list = []
        video_score_list = []

        ds_test.reset()
        for i in range(ds_test.batches):
            test_batch_x, test_batch_y = ds_test.next_batch()
            if print_test_loss:
                sum_loss, joint_loss, frame_loss, video_loss, frame_acc, video_acc \
                    = causalgraph.model.test_on_batch(x=test_batch_x, y=test_batch_y)
                print('Evaluation: Loss %g, Joint Loss %g, Frame Loss %g, Video Loss %g' %
                      (sum_loss, joint_loss, frame_loss, video_loss))
                # print('Evaluation: Frame Acc %g, Video Acc %g' % (frame_acc, video_acc))
            joint_label, frame_label, video_label = test_batch_y
            forgery_joint_label = joint_label == 2
            video_label_list.append(video_label)

            joint_score, frame_score, video_score = causalgraph.model.predict_on_batch(x=test_batch_x)
            forgery_joint_score = joint_score[:, :, -1]
            video_score_list.append(video_score)

            top_25 = range(1, 26)
            top_5 = range(1, 6)
            for j in range(len(joint_label)):
                # joint metrics
                lb_idx = forgery_joint_label[j]
                if sum(lb_idx) == 5:
                    normal_score = forgery_joint_score[j, lb_idx]
                    nScore = forgery_joint_score[j, ~lb_idx]
                    rec = recallAtK(normal_score, nScore, top_25)
                    prc = precisionAtK(normal_score, nScore, top_5)
                    # collect results
                    joint_rec_list.append(rec)
                    joint_prc_list.append(prc)

                # frame metrics
                frame_score_j = frame_score[j]
                frame_pred_score = frame_score_j[frame_label[j]]
                rank = np.sum(frame_pred_score < frame_score_j) + 1
                mrr = 1 / rank
                normal_score = frame_score_j[frame_label[j]]
                nScore = np.delete(frame_score_j, frame_label[j])
                r_rank = np.sum(frame_pred_score >= frame_score_j)
                frame_auc = (r_rank - 1)/(len(frame_score_j) - 1)
                # collect results
                frame_ar_list.append(rank)
                frame_arr_list.append(r_rank)
                frame_mrr_list.append(mrr)
                frame_auc_list.append(frame_auc)

            frame_pred_class = np.argmax(frame_score, axis=-1)
            # nsf_acc = accuracy_score(frame_label == 0, frame_pred_class == 0)
            nsf_acc = accuracy_score(frame_label, frame_pred_class)
            nsf_acc_list.append(nsf_acc)

            # video metrics
            video_pred_class = np.argmax(video_score, axis=-1)
            acc = accuracy_score(video_label > 0, video_pred_class > 0)
            acc_d = accuracy_score(video_label, video_pred_class)
            video_acc_list.append(acc)
            video_accd_list.append(acc_d)

        mean_joint_rec = np.mean(joint_rec_list, axis=0)
        mean_joint_prc = np.mean(joint_prc_list, axis=0)
        mean_frame_ar = np.mean(frame_ar_list)
        mean_frame_arr = np.mean(frame_arr_list)
        mean_frame_mrr = np.mean(frame_mrr_list)
        mean_frame_auc = np.mean(frame_auc_list)
        mean_nsf_acc = np.mean(nsf_acc_list)
        mean_video_acc = np.mean(video_acc_list)
        mean_video_accd = np.mean(video_accd_list)

        video_label_list = np.concatenate(video_label_list)
        video_score_list = np.concatenate(video_score_list)
        pos_idx = video_label_list == 0   # positive instances
        neg_s_idx = video_label_list == 1   # switch instances
        neg_r_idx = video_label_list == 2   # replace instances

        normal_score = 1 - video_score_list[pos_idx, 0]
        neg_s_score = 1 - video_score_list[neg_s_idx, 0]
        neg_r_score = 1 - video_score_list[neg_r_idx, 0]
        video_s_auc = auc(neg_s_score, normal_score)
        video_r_auc = auc(neg_r_score, normal_score)


        print('Joint metric: P@1: %g, P@5: %g, R@10: %g' % (mean_joint_prc[0], mean_joint_prc[4], mean_joint_rec[9]))
        print('Frame metric: NSF_ACC: %g, MRR: %g, AR: %g, ARR: %g, AUC: %g' %
              (mean_nsf_acc, mean_frame_mrr, mean_frame_ar, mean_frame_arr, mean_frame_auc))
        print('Video metric: ACC: %g, ACC_d: %g, AUC_S: %g, AUC_R: %g' % (mean_video_acc, mean_video_accd, video_s_auc, video_r_auc))

        print('==============================================================')

        results = {'mean_joint_rec': mean_joint_rec, 'mean_joint_prc': mean_joint_prc, 'mean_frame_ar': mean_frame_ar,
                   'mean_frame_mrr': mean_frame_mrr, 'mean_frame_auc': mean_frame_auc, 'mean_nsf_acc': mean_nsf_acc,
                   'mean_video_acc': mean_video_acc, 'video_accd_list': video_accd_list,
                   'joint_rec_list': joint_rec_list, 'joint_prc_list': joint_prc_list, 'frame_ar_list': frame_ar_list,
                   'frame_mrr_list': frame_mrr_list, 'frame_auc_list': frame_auc_list, 'nsf_acc_list': nsf_acc_list,
                   'video_acc_list': video_acc_list, 'mean_video_accd': mean_video_accd, 'video_s_auc': video_s_auc,
                   'video_r_auc':video_r_auc}
        filename = Path(__file__).stem
        io.savemat(filename + '.mat', results)

    epoch += 1
