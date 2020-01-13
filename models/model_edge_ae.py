""" TF model for point cloud autoencoder. PointNet encoder, FC decoder.
Using GPU Chamfer's distance loss.

Author: Charles R. Qi
Date: May 2018
"""
from itertools import permutations
import tensorflow as tf
import numpy as np
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


def init_edges_from_faces(faces):
    conn = faces
    all_edges = np.concatenate([conn[:, :2], conn[:,1:], np.concatenate([conn[:,0:1], conn[:,2:3]], 1)])
    min_edge, max_edge = np.min(all_edges, 1), np.max(all_edges, 1)
    edges = ["_".join(x) for x in zip(map(str,min_edge), map(str,max_edge))]
    unique_edges = list(set(edges))
    mesh_edges = np.array([x.split("_") for x in unique_edges]).astype(int)
    return mesh_edges, conn

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


def compute_edge_lengths(pred, faces=None):
    if faces is None:
        mesh_edges, _ = init_edges()
    else:
        mesh_edges, _ = init_edges_from_faces(faces)

    edges1 = tf.constant(mesh_edges[:,0], dtype = tf.int32)
    edges2 = tf.constant(mesh_edges[:,1], dtype = tf.int32)
    p1 =  tf.map_fn(lambda x : tf.gather(x,edges1), pred)
    p2 = tf.map_fn(lambda x : tf.gather(x,edges2), pred)
    edge_length = tf.math.sqrt(tf.maximum(tf.reduce_sum(tf.math.square(p1-p2), 2), 1e-12)) # TODO change here
    return edge_length


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
    end_points = {}

    input_image = tf.expand_dims(point_cloud, -1)

    # Encoder
    net = get_fc_encoder(input_image, is_training, bn_decay, num_point,batch_size, end_points)
    end_points['embedding'] = net

    # FC Decoder
    net = get_decoder(net, is_training, bn_decay, num_point, batch_size)

    return net, end_points



def get_fc_encoder(input_image, is_training, bn_decay, num_point,batch_size, end_points):
    input_image= tf.reshape(input_image, [batch_size, -1])
    net = tf_util.fully_connected(input_image, 1024, activation_fn=None, scope='enc_fc1')
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='enc_fc2', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='enc_fc3', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='enc_fc4', bn_decay=bn_decay)
    return net


def get_decoder(net, is_training, bn_decay, num_point, batch_size):
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, num_point, activation_fn=None, scope='fc3')
    return net


def get_pairs(batch_size):
    perm = list(permutations(range(batch_size), 2))
    perm = list(set([tuple(sorted(x)) for x in perm]))
    return tf.constant(perm)
