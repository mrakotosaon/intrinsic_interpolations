from itertools import permutations
import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
from scipy.io import loadmat

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

def compute_edge_lengths(pred):
    mesh_edges, _ = init_edges()
    edges1 = tf.constant(mesh_edges[:,0], dtype = tf.int32)
    edges2 = tf.constant(mesh_edges[:,1], dtype = tf.int32)
    #p1 =  tf.map_fn(lambda x : tf.gather(x,edges1), pred)
    #p2 = tf.map_fn(lambda x : tf.gather(x,edges2), pred)
    p1 = tf.gather(pred, edges1,axis = 1)
    p2 = tf.gather(pred, edges2,axis = 1)
    edge_length = tf.math.sqrt(tf.maximum(tf.reduce_sum(tf.math.square(p1-p2), 2), 1e-12)) # TODO change here
    return edge_length

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

def get_encoder(input_image, point_dim, is_training, bn_decay, num_point,batch_size, end_points):
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
    net = tf.identity(net, name = "old_input_latent")
    z = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='z', bn_decay=bn_decay)# 256
    return z


def get_model(point_cloud, is_training, bn_decay=None):
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
    net = get_encoder(input_image, point_dim, is_training, bn_decay, num_point,batch_size, end_points)
    end_points['embedding'] = net

    # FC Decoder
    net = get_decoder(net, is_training, bn_decay, num_point, batch_size)

    return net, end_points


def init_edges():
    conn_path = os.path.join(ROOT_DIR,"TRIV2.mat")
    conn = loadmat(conn_path)['TRIV']
    conn = conn-1
    all_edges = np.concatenate([conn[:, :2], conn[:,1:], np.concatenate([conn[:,0:1], conn[:,2:3]], 1)])
    min_edge, max_edge = np.min(all_edges, 1), np.max(all_edges, 1)
    edges = ["_".join(x) for x in zip(map(str,min_edge), map(str,max_edge))]
    unique_edges = list(set(edges))
    mesh_edges = np.array([x.split("_") for x in unique_edges]).astype(int)
    return mesh_edges, conn

def get_loss(pred, label, end_points):
    """ pred: BxNx3,
        label: BxNx3, """
    end_points['recons_loss'] = tf.reduce_mean(tf.square(label - pred))

def get_loss_ICP(pred, label, end_points):
    """ pred: BxNx3,
        label: BxNx3, """
    end_points['recons_loss'] = tf.reduce_mean(tf.square(ICP(label, pred, pred.shape[1]) - pred))


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
        loss = get_loss(outputs[0], tf.zeros((32,1024,3)), outputs[1])
