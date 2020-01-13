import tensorflow as tf
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    return pointclouds_pl, labels_pl

def get_decoder(net, is_training, bn_decay, num_point, batch_size):
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, num_point*3, activation_fn=None, scope='fc3')
    net = tf.reshape(net, (batch_size, num_point, 3))
    net = tf.identity(net, name = "pred")
    return net


def ICP(X, Y, n_pc_points):
    mu_x = tf.reduce_mean(X, axis = 1)
    mu_y =  tf.reduce_mean(Y, axis = 1)

    concat_mu_x = tf.tile(tf.expand_dims(mu_x,1), [1, n_pc_points, 1])
    concat_mu_y = tf.tile(tf.expand_dims(mu_y,1), [1, n_pc_points, 1])

    centered_y = tf.expand_dims(Y - concat_mu_y, 2)
    centered_x = tf.expand_dims(X - concat_mu_x, 2)

    # transpose y
    centered_y = tf.einsum('ijkl->ijlk', centered_y)

    mult_xy = tf.einsum('abij,abjk->abik', centered_y, centered_x)
    # sum
    C = tf.einsum('abij->aij', mult_xy)
    s, u,v = tf.svd(C)
    v = tf.einsum("aij->aji", v)

    R_opt = tf.einsum("aij,ajk->aik", u, v)#tf.matmul(u,v)

    t_opt = mu_y - tf.einsum("aki,ai->ak", R_opt, mu_x)
    concat_R_opt = tf.tile(tf.expand_dims(R_opt,1), [1, n_pc_points, 1, 1])
    concat_t_opt = tf.tile(tf.expand_dims(t_opt,1), [1, n_pc_points, 1])
    opt_labels =  tf.einsum("abki,abi->abk", concat_R_opt, X) + concat_t_opt
    return opt_labels

def get_encoder_vae(input_image, point_dim, is_training, bn_decay, num_point,batch_size, end_points):
    net = tf_util.conv2d(input_image, 64, [1,point_dim],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    point_feat = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(point_feat, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1,1],# 512
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    global_feat = tf_util.max_pool2d(net, [num_point,1],
                                     padding='VALID', scope='maxpool')

    net = tf.reshape(global_feat, [batch_size, -1])
    z = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='z', bn_decay=bn_decay)# 256
    z= tf.identity(z, name = "old_input_latent")

    z_mean = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='mean', bn_decay=bn_decay)# 256
    z_std = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='std', bn_decay=bn_decay)# 256
    eps = tf.random_normal(shape=tf.shape(z_std), mean=0, stddev=1, dtype=tf.float32)
    z = z_mean + z_std * eps
    z = tf.identity(z, name = "input_latent")
    kl_divergence =  0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(z_mean) + tf.square(z_std) - tf.log(1e-10 +tf.square(z_std)) - 1, 1))
    end_points['kl_div'] = kl_divergence
    return z


def get_model_vae(point_cloud, is_training, bn_decay=None):
    """ Autoencoder for point clouds.
    Input:
        point_cloud: TF tensor BxNx3
        is_training: boolean
        bn_decay: float between 0 and 1
    Output:
        net: TF tensor BxNx3, reconstructed point clouds
        end_points: dict
    """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    point_dim = point_cloud.get_shape()[2].value
    end_points = {}

    input_image = tf.expand_dims(point_cloud, -1)

    # Encoder
    net = get_encoder_vae(input_image, point_dim, is_training, bn_decay, num_point,batch_size, end_points)
    end_points['embedding'] = net

    # FC Decoder
    net = get_decoder(net, is_training, bn_decay, num_point, batch_size)

    return net, end_points



def get_loss(pred, label, end_points):
    """ pred: BxNx3,
        label: BxNx3, """
    end_points['recons_loss'] = tf.reduce_mean(tf.square(label - pred))

def get_loss_ICP(pred, label, end_points):
    """ pred: BxNx3,
        label: BxNx3, """
    end_points['recons_loss'] = tf.reduce_mean(tf.square(ICP(label, pred, pred.shape[1]) - pred))
