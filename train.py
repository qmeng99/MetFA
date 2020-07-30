import os
import tensorflow as tf
import numpy as np
import residual_def
import pdb
import random


height = 224
width = 288
batch_size = 6
lr = 1e-4
model_dir = './model'
logs_path = './model'
max_iter_step = 30010
anat_num = 6
func_num = 2
seed = 42
val_each_num = 5
val_imgnum = anat_num*val_each_num
latentdim = 8*28*36


def read_decode(filename_queue, minibatch):
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
        features={"img": tf.FixedLenFeature([],tf.string),
                  "anatlbl": tf.FixedLenFeature([], tf.int64),
                  "funclbl": tf.FixedLenFeature([], tf.int64)})
    image = tf.decode_raw(features["img"], tf.float32)
    Anat_label = tf.cast(features["anatlbl"], tf.int64)
    Func_label = tf.cast(features["funclbl"], tf.int64)
    image = tf.reshape(image, [height, width, 1])
    images, Anat_labels, Func_labels = tf.train.batch([image, Anat_label, Func_label], batch_size=minibatch, capacity=1000, num_threads=8)
    # images and labels are tensor object
    return images, Anat_labels, Func_labels

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

def load():

    # load data to keep the balance in each big batch, the batch size here is the minibatch
    # load labelled data

    filename_1 = '/data/train/4CH_1.tfrecords'
    filename_queue_1 = tf.train.string_input_producer([filename_1])
    image_1, anat_lbl_1, func_lbl_1 = read_decode(filename_queue_1, batch_size)

    filename_2 = '/data/train/Abdominal_1.tfrecords'
    filename_queue_2 = tf.train.string_input_producer([filename_2])
    image_2, anat_lbl_2, func_lbl_2 = read_decode(filename_queue_2, batch_size)

    filename_3 = '/data/train/LVOT_1.tfrecords'
    filename_queue_3 = tf.train.string_input_producer([filename_3])
    image_3, anat_lbl_3, func_lbl_3 = read_decode(filename_queue_3, batch_size)

    filename_4 = '/data/train/RVOT_1.tfrecords'
    filename_queue_4 = tf.train.string_input_producer([filename_4])
    image_4, anat_lbl_4, func_lbl_4 = read_decode(filename_queue_4, batch_size)

    filename_5 = '/data/train/Lips_1.tfrecords'
    filename_queue_5 = tf.train.string_input_producer([filename_5])
    image_5, anat_lbl_5, func_lbl_5 = read_decode(filename_queue_5, batch_size)

    filename_6 = '/data/train/Femur_1.tfrecords'
    filename_queue_6 = tf.train.string_input_producer([filename_6])
    image_6, anat_lbl_6, func_lbl_6 = read_decode(filename_queue_6, batch_size)


    # load unlabelled data
    filename_U_0 = 'data/train/train_unlabelled.tfrecords'
    filename_queue_U_0 = tf.train.string_input_producer([filename_U_0])
    image_U_0, anat_lbl_U_0, func_lbl_U_0 = read_decode(filename_queue_U_0, 6*batch_size)

    # data
    image = tf.concat([image_1, image_2, image_3, image_4, image_5, image_6], 0)
    anatlbl_anat = tf.concat(
        [anat_lbl_1, anat_lbl_2, anat_lbl_3, anat_lbl_4, anat_lbl_5, anat_lbl_6], 0)

    funclbl_func = tf.concat([func_lbl_1, func_lbl_2,
                              func_lbl_3, func_lbl_4, func_lbl_5, func_lbl_6], 0)

    print (image.shape, anatlbl_anat.shape, funclbl_func.shape)

    return image, anatlbl_anat, funclbl_func, image_U_0, anat_lbl_U_0, func_lbl_U_0

def predictedcount(fea, logit, prob, flag=1):

    # the output is batchsize*classnum. the position that satisfied the condition is 1, if no condition is
    # satisfied then all the matrix is 0

    #flag is to show whether it is source image or target image, flag==0 is source, flag==1 is target

    if flag == 0:
        y = prob*1.0
    elif flag == 1:
        y_argmax = tf.argmax(prob, axis=1)
        y = tf.one_hot(y_argmax, depth=anat_num)
        # y = tf.cast(tf.greater_equal(prob, 0.5), tf.float32)

    print ('predictedcount function:')
    print (fea.shape, logit.shape, prob.shape)

    comb_fea = tf.einsum('ij,jk->ik', tf.transpose(y), fea)

    print (comb_fea.shape)

    comb_logit = tf.einsum('ij,jk->ik', tf.transpose(y), logit)

    print (comb_logit.shape)

    y_sum_new = tf.reduce_sum(y, axis=0, keep_dims=True)

    print (y_sum_new.shape)

    mask = tf.cast(tf.greater_equal(y_sum_new, 1e-8), tf.float32)

    print (mask.shape)

    y_sum_revise = y_sum_new + 1e-10   # to avoid divided by 0

    comb_fea_mean = comb_fea / tf.transpose(y_sum_revise)  #shape=classnum*featuredims
    comb_logit_mean = comb_logit / tf.transpose(y_sum_revise)  # shape=classnum*predictions

    print (comb_fea_mean.shape, comb_logit_mean.shape)

    return comb_fea_mean, comb_logit_mean, mask

def KLD(logit1, logit2, mask, temperature=2.0):

    # logit1 from target domain and logit2 from source

    prob1 = tf.nn.softmax(logit1 / temperature)
    prob2 = tf.nn.softmax(logit2 / temperature)

    prob1 = tf.clip_by_value(prob1, clip_value_min=1e-8, clip_value_max=1.0)  # for preventing prob=0 resulting in NAN
    prob2 = tf.clip_by_value(prob2, clip_value_min=1e-8, clip_value_max=1.0)

    print ('KLD function')
    print (prob1.shape, prob2.shape)

    # KL_div = (tf.reduce_sum(prob1 * tf.log(prob1 / prob2)) + tf.reduce_sum(prob2 * tf.log(prob2 / prob1))) / 2.0
    KL_distance = 0.5 * (prob1 * tf.log(prob1 / prob2) + prob2 * tf.log(prob2 / prob1))
    KL_div = tf.reduce_sum(tf.einsum('ij,jk->ik', mask, KL_distance))

    num_sample = tf.reduce_sum(mask)

    kd_loss = KL_div / num_sample

    return kd_loss, prob1, prob2

def separateset(fea):

    print ('separateset function')
    print (fea.shape)

    # separate support set and query set
    queryset = tf.expand_dims(fea[0:1, :], axis=0)
    supportset = tf.expand_dims(fea[1: batch_size, :], axis=0)
    for i in range(1, anat_num):
        querytemp = tf.expand_dims(fea[i * batch_size:i * batch_size + 1, :], axis=0)
        queryset = tf.concat([queryset, querytemp], axis=0)

        supporttemp = tf.expand_dims(fea[i * batch_size + 1:(i+1) * batch_size, :], axis=0)
        supportset = tf.concat([supportset, supporttemp], axis=0)


    print (queryset.shape, supportset.shape)

    return queryset, supportset


def transML(S_fea, T_fea, T_prob, t):

    _, supportset = separateset(S_fea)

    y_argmax = tf.cast(tf.argmax(T_prob, axis=1), tf.int32)

    print ('transML function')
    print (y_argmax.shape)

    loss = 0.0
    for i in range(anat_num*batch_size):
        querysample = T_fea[i:i+1, :]
        querysample_exd = tf.expand_dims(querysample, axis=0)

        print (supportset.shape, querysample_exd.shape)

        # shape should be classnum*support number
        querycelldist = tf.reduce_sum(tf.sqrt(tf.pow(tf.subtract(supportset, querysample_exd) + 1e-8, 2)), axis=2)

        print (querycelldist.shape)

        mindist = -tf.reduce_min(querycelldist, axis=1, keep_dims=True)  # the shape should be classnum*1
        maxdist = -tf.reduce_max(querycelldist, axis=1, keep_dims=True)  # the shape should be classnum*1

        mask_1 = tf.Variable(tf.ones(shape=(anat_num, 1), dtype=tf.float32))
        for k in range(anat_num):
            if y_argmax[i] == k:
                mask_1 = tf.assign(mask_1[k,0], 0)

        dist = mindist * mask_1 + maxdist * (1 - mask_1)

        querylossvector = tf.nn.log_softmax(tf.transpose(dist))  # tf.nn.log_softmax can only work with 1*dim vector

        queryloss = (-1.0) * (querylossvector[0, y_argmax[i]])

        loss = loss + queryloss

    loss = loss / (anat_num * batch_size)

    return loss

def build_gpu():

    with tf.device("/gpu:0"):

        image_S_orig, anat_lbl_S, func_lbl_S, image_T_orig, anat_lbl_T, func_lbl_T = load()

        # data augmentation: adding noise
        image_S_noise = gaussian_noise_layer(image_S_orig, 0.1)
        image_T_noise = gaussian_noise_layer(image_T_orig, 0.1)
        # data augmentation: random flip
        image_S_squz = tf.transpose(tf.squeeze(image_S_noise, axis=3), [1, 2, 0])
        image_S_flip = tf.image.random_flip_left_right(image_S_squz)
        image_S_flip = tf.expand_dims(tf.transpose(image_S_flip, [2, 0, 1]), axis=3)
        image_S = tf.expand_dims(image_S_flip, axis=3)

        image_T_squz = tf.transpose(tf.squeeze(image_T_noise, axis=3), [1, 2, 0])
        image_T_flip = tf.image.random_flip_left_right(image_T_squz)
        image_T_flip = tf.expand_dims(tf.transpose(image_T_flip, [2, 0, 1]), axis=3)
        image_T = tf.expand_dims(image_T_flip, axis=3)

        w_cls = tf.Variable(10, dtype=tf.float32, trainable=False)
        w_latent = tf.Variable(1e-2, dtype=tf.float32, trainable=False)
        w_mme = tf.Variable(5, dtype=tf.float32, trainable=False)
        w_ml = tf.Variable(0, dtype=tf.float32, trainable=False)
        w_trans = tf.Variable(1e-1, dtype=tf.float32, trainable=False)
        w_rec = tf.Variable(1, dtype=tf.float32, trainable=False)
        w_kd = tf.Variable(0, dtype=tf.float32, trainable=False)
        l_r = tf.Variable(lr, dtype=tf.float32, trainable=False)

        # opt_adv = tf.train.AdamOptimizer(learning_rate=l_r, beta1=0., beta2=0.9, epsilon=1e-5)
        # opt_adv = tf.train.MomentumOptimizer(learning_rate=l_r, momentum=0.9)
        opt_cls = tf.train.MomentumOptimizer(learning_rate=l_r, momentum=0.9)

        # ----------------------Encoder-------------------------

        with tf.variable_scope('Encoder_S'):
            S_fea, S_res_scales, S_saved_strides, S_filters = residual_def.residual_encoder(
                inputs=image_S,
                num_res_units=1,
                mode=tf.estimator.ModeKeys.TRAIN,
                filters=(8, 16, 32, 64, 8),
                strides=((1, 1, 1), (2, 2, 1), (2, 2, 1), (2, 2, 1), (1, 1, 1)),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

            S_fea_flatten = tf.contrib.layers.flatten(S_fea)

        with tf.variable_scope('VAE_mu'):
            z_S_mu = residual_def.VAE_layer(x=S_fea_flatten,
                                            outputdim=4096,
                                            is_train=True,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))
        with tf.variable_scope('VAE_sigma'):
            z_S_log_sigma_sq = residual_def.VAE_layer(x=S_fea_flatten,
                                            outputdim=4096,
                                            is_train=True,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

        eps_S = tf.random_normal(
            shape=tf.shape(z_S_log_sigma_sq),
            mean=0, stddev=1, dtype=tf.float32)
        z_S = z_S_mu + tf.multiply(tf.sqrt(tf.exp(z_S_log_sigma_sq)), eps_S)

        # ----------------------Encoder-------------------------

        with tf.variable_scope('Encoder_S', reuse=True):
            T_fea, T_res_scales, T_saved_strides, T_filters = residual_def.residual_encoder(
                inputs=image_T,
                num_res_units=1,
                mode=tf.estimator.ModeKeys.TRAIN,
                filters=(8, 16, 32, 64, 8),
                strides=((1, 1, 1), (2, 2, 1), (2, 2, 1), (2, 2, 1), (1, 1, 1)),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

            T_fea_flatten = tf.contrib.layers.flatten(T_fea)

        with tf.variable_scope('VAE_mu', reuse=True):
            z_T_mu = residual_def.VAE_layer(x=T_fea_flatten,
                                            outputdim=4096,
                                            is_train=True,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))
        with tf.variable_scope('VAE_sigma', reuse=True):
            z_T_log_sigma_sq = residual_def.VAE_layer(x=T_fea_flatten,
                                            outputdim=4096,
                                            is_train=True,
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

        eps_T = tf.random_normal(
            shape=tf.shape(z_T_log_sigma_sq),
            mean=0, stddev=1, dtype=tf.float32)
        z_T = z_T_mu + tf.multiply(tf.sqrt(tf.exp(z_T_log_sigma_sq)), eps_T)


        # ----------------------for transfered fea num_classification----------------------

        with tf.variable_scope('anat_cls'):
            anat_new_S = residual_def.classify_dense_bn_relu(
                z_S,
                units=(512, 128),
                is_train=True,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

        with tf.variable_scope('prototype'):
            anat_logits_S = residual_def.prototype(
                anat_new_S,
                is_train=True,
                num_class=anat_num,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

        # ----------------------num_prediction for unlabled data----------------------

        with tf.variable_scope('anat_cls', reuse=True):
            anat_new_T = residual_def.classify_dense_bn_relu(
                z_T,
                units=(512, 128),
                is_train=True,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

        with tf.variable_scope('prototype', reuse=True):
            anat_logits_T = residual_def.prototype(
                anat_new_T,
                is_train=True,
                num_class=anat_num,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))


        # ----------------------Source reconstruction----------------------

        # Here we didn't use skip connection, see upsample and rescale

        with tf.variable_scope('S_reconstruction'):

            z_S_reshape = tf.layers.dense(inputs=z_S, units=latentdim, trainable=True)


        with tf.variable_scope('T_reconstruction'):
            z_T_reshape = tf.layers.dense(inputs=z_T, units=latentdim, trainable=True)


        # print out the shape of above outputs
        print (S_fea.shape, T_fea.shape, anat_logits_S.shape, anat_logits_T.shape)

        # ----------------------classification Loss--------------------------

        # cross entropy loss of labeled data
        labels_onehot_anat = tf.one_hot(anat_lbl_S, depth=anat_num)
        anat_cls_loss_labeled = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=anat_logits_S, labels=labels_onehot_anat))
        reg_anat = tf.losses.get_regularization_loss('anat_cls')
        anat_cls_loss = anat_cls_loss_labeled + reg_anat

        # entropy loss of unlabeled data
        predictlabel_T = tf.nn.softmax(anat_logits_T)
        adloss = -tf.reduce_mean(tf.reduce_sum(predictlabel_T * (tf.log(predictlabel_T + 1e-8)), 1))

        # -----------------------VAE loss---------------------------

        # Latent loss
        # KL divergence: measure the difference between two distributions
        # Here we measure the divergence between
        # the latent distribution and N(0, 1)

        S_latent_loss = -0.5 * tf.reduce_sum(
            1 + z_S_log_sigma_sq - tf.square(z_S_mu) -
            tf.exp(z_S_log_sigma_sq), axis=1)

        T_latent_loss = -0.5 * tf.reduce_sum(
            1 + z_T_log_sigma_sq - tf.square(z_T_mu) -
            tf.exp(z_T_log_sigma_sq), axis=1)

        latent_loss_S = tf.reduce_mean(S_latent_loss)
        latent_loss_T = tf.reduce_mean(T_latent_loss)
        latent_loss = 0.5 * (latent_loss_S + latent_loss_T)


  
        # -------------------------transML loss-----------------------------------------
        predictlabel_T = tf.nn.softmax(anat_logits_T)
        trans_loss = transML(z_S, z_T, predictlabel_T, t=0.2)

        # -------------------------reconstrcuction loss-----------------------------------------

        S_lossrecon = tf.reduce_mean(tf.pow(tf.subtract(z_S_reshape, S_fea_flatten), 2))
        S_reg_recon = tf.losses.get_regularization_loss('S_reconstruction')
        T_lossrecon = tf.reduce_mean(tf.pow(tf.subtract(z_T_reshape, T_fea_flatten), 2))
        T_reg_recon = tf.losses.get_regularization_loss('T_reconstruction')

        recon_loss = 0.5 * (S_lossrecon + S_reg_recon + T_lossrecon + T_reg_recon)

        # ------------------------distillation loss-------------------------------------------

        # predict the label of unlabeled target domain data

        # selec_mask shows the effective sample, e.g. selec_mask=[0,0,1] means the third sample is the valid one, and will be used latter for KD, FD
        # mask is 1*classnumdim, tf.reduce_sum(mask) computes the number of valid samples, because the maks is a binary vector
        pred_aver_feature, pred_aver_logit, pred_mask = predictedcount(T_fea_flatten, anat_logits_T, predictlabel_T,
                                                                       flag=1)  # the output has the classnum*dims shape
        S_aver_feature, S_aver_logit, S_mask = predictedcount(S_fea_flatten, anat_logits_S, labels_onehot_anat, flag=0)

        KD_loss, p1, p2 = KLD(pred_aver_logit, S_aver_logit, pred_mask, temperature=2)

        # -----------------------Total loss
        loss_all = w_cls * anat_cls_loss + w_latent * latent_loss + w_trans * trans_loss + w_rec * recon_loss
        loss_cls = loss_all - w_mme * adloss
        loss_enc = loss_all + w_mme * adloss


        # ------------------optimization----------------------------
        Encoder_S_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Encoder_S')
        anat_cls_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'anat_cls')
        VAE_mu_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'VAE_mu')
        VAE_sigma_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'VAE_sigma')
        prototype_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'prototype')
        S_rec_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'S_reconstruction')
        T_rec_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'T_reconstruction')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = opt_cls.minimize(loss_all,
                                        var_list=[Encoder_S_var, anat_cls_var, VAE_mu_var, VAE_sigma_var, prototype_var,
                                                  S_rec_var, T_rec_var])
            trian_cls_op = opt_cls.minimize(loss_cls, var_list=prototype_var)
            trian_enc_op = opt_cls.minimize(loss_enc, var_list=[Encoder_S_var, anat_cls_var, VAE_mu_var, VAE_sigma_var])

        weight_decay = 5e-4
        with tf.control_dependencies([train_op]):
            l2_loss_1 = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in Encoder_S_var])
            l2_loss_2 = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in anat_cls_var])
            l2_loss_3 = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in VAE_mu_var])
            l2_loss_4 = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in VAE_sigma_var])
            l2_loss_5 = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in prototype_var])
            l2_loss_6 = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in S_rec_var])
            l2_loss_7 = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in T_rec_var])
            sgd = tf.train.GradientDescentOptimizer(learning_rate=1.0)
            decay_op = sgd.minimize(l2_loss_1 + l2_loss_2 + l2_loss_3 + l2_loss_4 + l2_loss_5 + l2_loss_6 + l2_loss_7)


    return train_op, trian_cls_op, trian_enc_op, decay_op, \
           anat_cls_loss, adloss, latent_loss, trans_loss, recon_loss, KD_loss, \
           w_cls, w_latent, w_ml, w_mme, w_trans, w_rec, w_kd, \
           image_S, image_T, latent_loss_S, latent_loss_T, losstemp

def main():
    train_op, trian_cls_op, trian_enc_op, decay_op, \
    anat_cls_loss, adloss, latent_loss, trans_loss, recon_loss, KD_loss, \
    w_cls, w_latent, w_ml, w_mme, w_trans, w_rec, w_kd, \
    image_S, image_T, latent_loss_S, latent_loss_T, losstemp = build_gpu()

    # ----------------validation---------------------------------
    image_orig = tf.placeholder(dtype=tf.float32, shape=[val_imgnum, height, width, 1])
    lblanat = tf.placeholder(dtype=tf.int64, shape=[val_imgnum])

    # ----------------------Encoder-------------------------
    image_val = tf.expand_dims(image_orig, 3)

    # ----------------------Encoder-------------------------

    with tf.variable_scope('Encoder_S', reuse=True):
        T_fea_val, T_res_scales_val, T_saved_strides_val, T_filters_val = residual_def.residual_encoder(
            inputs=image_val,
            num_res_units=1,
            mode=tf.estimator.ModeKeys.EVAL,
            filters=(8, 16, 32, 64, 8),
            strides=((1, 1, 1), (2, 2, 1), (2, 2, 1), (2, 2, 1), (1, 1, 1)),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

        T_fea_flatten_val = tf.contrib.layers.flatten(T_fea_val)

    with tf.variable_scope('VAE_mu', reuse=True):
        z_T_mu_val = residual_def.VAE_layer(x=T_fea_flatten_val,
                                        outputdim=4096,
                                        is_train=False,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))
    with tf.variable_scope('VAE_sigma', reuse=True):
        z_T_log_sigma_sq_val = residual_def.VAE_layer(x=T_fea_flatten_val,
                                        outputdim=4096,
                                        is_train=False,
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))


    z_T_val = z_T_mu_val

    with tf.variable_scope('anat_cls', reuse=True):
        anat_new_T_val = residual_def.classify_dense_bn_relu(
            z_T_val,
            units=(512, 128),
            is_train=False,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    with tf.variable_scope('prototype', reuse=True):
        anat_logits_T_val = residual_def.prototype(
            anat_new_T_val,
            is_train=False,
            num_class=anat_num,
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-5))

    # ----------------------Loss--------------------------
    onehot_anat = tf.one_hot(lblanat, depth=anat_num)
    anat_cls_loss_val = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=anat_logits_T_val, labels=onehot_anat))

    print ('val dimention')
    print (anat_logits_T_val.shape)

    loss_val = anat_cls_loss_val

    val_anat_label = tf.argmax(tf.nn.softmax(anat_logits_T_val), axis=1)

    # -----------------------------------------------------------

    saver = tf.train.Saver(max_to_keep=5)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    tf.set_random_seed(seed)
    np.random.seed(seed)

    with tf.Session(config=config) as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        # Create a summary to monitor cost tensor
        tf.summary.scalar("anat_cls_loss", anat_cls_loss)
        tf.summary.scalar("MME_loss", adloss)
        tf.summary.scalar("Latent_loss", latent_loss)
        tf.summary.scalar("Trans_loss", trans_loss)
        tf.summary.scalar("Recon_loss", recon_loss)
        tf.summary.scalar("KD_loss", KD_loss)


        tf.summary.scalar("latent_loss_S", latent_loss_S)
        tf.summary.scalar("latent_loss_T", latent_loss_T)

        tf.summary.scalar("loss_val", loss_val)


        tf.summary.image('image_S', image_S[:,:,:,:,0], tf.float32)
        tf.summary.image('image_T', image_T[:,:,:,:,0], tf.float32)

        tf.summary.image('image_val', image_val[:,:,:,:,0], tf.float32)

        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        sess.run(init_op)
        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)


            _, _, summary = sess.run([train_op, decay_op, merged_summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, i)
            _, _, summary = sess.run([trian_cls_op, decay_op, merged_summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, i)
            _, _, summary = sess.run([trian_enc_op, decay_op, merged_summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, i)

            anatclsLoss, ADLoss, latentLoss, transLoss, reconLoss, Losstemp, KDLoss = sess.run(
                [anat_cls_loss, adloss, latent_loss, trans_loss, recon_loss, losstemp, KD_loss])


            if i % 100 == 0:
                print("i = %d" % i)
                print ("Anat Cls Loss = {}".format(anatclsLoss))
                print ('MME loss = {}'.format(ADLoss))
                print ('Latent loss = {}'.format(latentLoss))
                print ('Trans loss = {}'.format(transLoss))
                print ('Recon loss = {}'.format(reconLoss))
                print ('KD loss = {}'.format(KDLoss))

            if i % 500 == 0:
                saver.save(sess, os.path.join(model_dir, "model.val"), global_step=i)

        coord.request_stop()
        coord.join(threads)

main()
















