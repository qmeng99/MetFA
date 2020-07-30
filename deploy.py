import os
import tensorflow as tf
import numpy as np
import residual_def
import pdb
import random
from sklearn.metrics import precision_recall_fscore_support, recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt


height = 224
width = 288
ckpt_dir = './model'
anat_num = 6
latentdim = 8*28*36



def load_data(flag):
    if flag == 0:
        testpath_1 = '/data/test/4CH_1.npz'
    else:
        testpath_1 = '/data/test/4CH_2.npz'
    testimg_1 = np.load(testpath_1)['img']
    test_lbl_anat_1 = np.load(testpath_1)['anatlbl']

    if flag == 0:
        testpath_2 = '/data/test/Abdominal_1.npz'
    else:
        testpath_2 = '/data/test/Abdominal_2.npz'
    testimg_2 = np.load(testpath_2)['img']
    test_lbl_anat_2 = np.load(testpath_2)['anatlbl']

    if flag == 0:
        testpath_3 = '/data/test/LVOT_1.npz'
    else:
        testpath_3 = '/data/test/LVOT_2.npz'
    testimg_3 = np.load(testpath_3)['img']
    test_lbl_anat_3 = np.load(testpath_3)['anatlbl']

    if flag == 0:
        testpath_4 = '/data/test/RVOT_1.npz'
    else:
        testpath_4 = '/data/test/RVOT_2.npz'
    testimg_4 = np.load(testpath_4)['img']
    test_lbl_anat_4 = np.load(testpath_4)['anatlbl']

    if flag == 0:
        testpath_5 = '/data/test/Lips_1.npz'
    else:
        testpath_5 = '/data/test/Lips_2.npz'
    testimg_5 = np.load(testpath_5)['img']
    test_lbl_anat_5 = np.load(testpath_5)['anatlbl']

    if flag == 0:
        testpath_6 = '/data/test/Femur_1.npz'
    else:
        testpath_6 = '/data/test/Femur_2.npz'
    testimg_6 = np.load(testpath_6)['img']
    test_lbl_anat_6 = np.load(testpath_6)['anatlbl']



    testimg = np.concatenate([testimg_1,testimg_2,testimg_3,testimg_4,testimg_5,testimg_6], axis=0)
    test_lbl_anat = np.concatenate([test_lbl_anat_1, test_lbl_anat_2, test_lbl_anat_3, test_lbl_anat_4, test_lbl_anat_5,
                                   test_lbl_anat_6], axis=0)


    return testimg, test_lbl_anat


def main():
    graph = tf.Graph()
    with graph.as_default():
        with tf.device("/gpu:0"):
            image_orig = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 1])
            lblanat = tf.placeholder(dtype=tf.int64, shape=[None])
            flag = tf.placeholder(dtype=tf.int64, shape=())

            image_val = tf.expand_dims(image_orig, axis=3)

            # ----------------------Encoder-------------------------

            with tf.variable_scope('Encoder_S'):
                T_fea, T_res_scales, T_saved_strides, T_filters = residual_def.residual_encoder(
                    inputs=image_val,
                    num_res_units=1,
                    mode=tf.estimator.ModeKeys.EVAL,
                    filters=(8, 16, 32, 64, 8),
                    strides=((1, 1, 1), (2, 2, 1), (2, 2, 1), (2, 2, 1), (1, 1, 1)),
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

                fea_new = tf.contrib.layers.flatten(T_fea)

            with tf.variable_scope('VAE_mu'):
                z_T_mu = residual_def.VAE_layer(x=fea_new,
                                                outputdim=4096,
                                                is_train=False,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

            z_T = z_T_mu


            # ----------------------num_classification----------------------

            with tf.variable_scope('anat_cls'):
                anat_new = residual_def.classify_dense_bn_relu(
                    z_T,
                    units=(512, 128),
                    is_train=False,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

            with tf.variable_scope('prototype'):
                anat_logits = residual_def.prototype(
                    anat_new,
                    is_train=False,
                    num_class=anat_num,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))


            test_anat_softmax = tf.nn.softmax(anat_logits)
            test_anat_label = tf.argmax(test_anat_softmax, axis=1)
            loss_anat_softce = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=anat_logits,
                                                        labels=tf.one_hot(lblanat, depth=anat_num)))

        # ---------------------------------------------------
        config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))

            # since the graph can't contain if/else, so the the deploy script will be separated
            # this script is only used for source domain data, so the flat MUST be 1
            STflag = 1  # use this flag to see it is source or target data

            data, lbl_anat = load_data(flag=STflag)
            data_new = np.reshape(data, (data.shape[0], height, width, 1))

            t_data = data_new
            t_anat_lbl = lbl_anat
            feed_dict = {image_orig: t_data, lblanat: t_anat_lbl, flag: STflag}

            prob_anat, pred_anat, loss_soft_anat, fea_T, fea_T_last = sess.run(
                [test_anat_softmax, test_anat_label, loss_anat_softce, z_T, anat_logits],
                feed_dict=feed_dict)



            # overall accuracy
            accuracy = accuracy_score(t_anat_lbl, pred_anat)
            print ("Overall Accuracy Anat= {:.4f}".format(accuracy))

            # precision, recall, f1score
            precision, recall, f1score, _ = precision_recall_fscore_support(t_anat_lbl, pred_anat, average=None,
                                                                            labels=[0, 1, 2, 3, 4, 5])

            # accuracy of each class and print the other measurement
            for i in range(0,6):

                right = 0
                index = np.where(t_anat_lbl == i)[0]

                y_true = t_anat_lbl[index]
                y_pred = pred_anat[index]

                for ss in range(len(index)):
                    if (y_true[ss] == i) and (y_pred[ss] == i):
                        right = right + 1

                # print ("Label {} accuracy= {:.4f}, precision= {:.4f}, recall= {:.4f}, f1score= {:.4f}, img_num={}".format
                #        (i, (right / len(index)), precision[i], recall[i], f1score[i], len(index)))
                print ("{:.4f}, {:.4f}, {:.4f}".format
                       (precision[i], recall[i], f1score[i]))

    

    return

main()
















