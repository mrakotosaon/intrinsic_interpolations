import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
from plyfile import PlyData, PlyElement
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
import part_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log_mapping2_losses', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1000, help='Point Number [default: 1000]')
parser.add_argument('--max_epoch', type=int, default=2501, help='Epoch to run [default: 201]')
parser.add_argument('--best_edge_ae_epoch', type=int, default=3458, help='Epoch to run [default: 201]')
parser.add_argument('--best_shape_vae_epoch', type=int, default=2291, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=466*32*200, help='Decay step for lr decay [default: 200000]')# 1000000
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--no_rotation', action='store_true', help='Disable random rotation during training.')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate


MODEL = importlib.import_module("model_edge_ae") # import network module
MODEL2 = importlib.import_module("model_shape_vae") # import network module

MODEL_FILE = os.path.join(BASE_DIR, 'model_edge_ae.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.1
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP =466*32*200
BN_DECAY_CLIP = 0.99
HOSTNAME = socket.gethostname()

DATA_PATH = "data/sliced"
EDGES_PATH = "data/template.ply"

TRAIN_DATASET = part_dataset.Dataset(root=DATA_PATH, npoints=NUM_POINT,  split='train')
TEST_DATASET = part_dataset.Dataset(root=DATA_PATH, npoints=NUM_POINT, split='val')

MODEL_PATH = "log_edge_ae/best_model_epoch_{}.ckpt".format(FLAGS.best_edge_ae_epoch)
MODEL_PATH2 = "log_shape_vae/best_model_epoch_{}.ckpt".format(FLAGS.best_shape_vae_epoch)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate_base = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate_base, 0.000001) # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay =tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay



def train():
    with tf.Graph().as_default() as g:
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            is_training_false = tf.constant(False)
            latent_pl = tf.placeholder(tf.float32, shape=(1, 256))
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            def to_pc_translator(net):
                net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training_pl, scope='to_pc_1', bn_decay=None)
                net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training_pl, scope='to_pc_2', bn_decay=None)
                net = tf_util.fully_connected(net, 256, activation_fn=None, scope='to_pc_3')
                return net

            def to_edge_translator(net):
                net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training_pl, scope='to_edge_1', bn_decay=None)
                net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training_pl, scope='to_edge_2', bn_decay=None)
                net = tf_util.fully_connected(net, 256, activation_fn=None, scope='to_edge_3')
                return net

            plydata = PlyData.read(EDGES_PATH)
            faces = plydata['face']
            faces = np.array([faces[i][0] for i in range(faces.count)])

            # cycle through shape_vae from edge_ae
            with tf.variable_scope("edge_ae"):
                label_edge_lengths = MODEL.compute_edge_lengths(labels_pl, faces)
                end_points = {}
                latent_repr_edge = MODEL.get_fc_encoder(tf.expand_dims(label_edge_lengths, -1),  is_training_false, None, label_edge_lengths.shape[1],32, end_points)
            with tf.variable_scope("to_pc"):
                from_edge_latent0= to_pc_translator(latent_repr_edge)
            with tf.variable_scope("to_edge"):
                from_pc_latent0 = to_edge_translator(from_edge_latent0)
            with tf.variable_scope("edge_ae"):
                edge_pred = MODEL.get_decoder(from_pc_latent0, is_training_false, None, label_edge_lengths.shape[1], BATCH_SIZE)


            # cycle through edge_ae from shape_vae
            with tf.variable_scope("reconstruction_pc"):
                latent_repr_pc = MODEL2.get_encoder_vae(tf.expand_dims(labels_pl, -1), 3, is_training_false, None, NUM_POINT,32, end_points)
            with tf.variable_scope("to_edge"):
                from_pc_latent1 = to_edge_translator(latent_repr_pc)
            with tf.variable_scope("to_pc"):
                from_edge_latent1 = to_pc_translator(from_pc_latent1)
            with tf.variable_scope("reconstruction_pc"):
                pc_recons = MODEL2.get_decoder(from_edge_latent1, is_training_false, None, NUM_POINT, BATCH_SIZE)
                edge_pc = MODEL.compute_edge_lengths(pc_recons, faces)

            diff_pc =  tf.reduce_mean(tf.square(pc_recons - MODEL2.ICP(labels_pl, pc_recons, NUM_POINT)))
            diff_edge_from_pc =  tf.reduce_mean(tf.square(label_edge_lengths - edge_pc))
            diff_edge =  tf.reduce_mean(tf.square(label_edge_lengths - edge_pred))

            recons_loss_alpha = 100.0
            diff_edge_alpha = 800.0
            diff_edge_from_pc_alpha = 2000.0

            tf.summary.scalar('recons_pc', diff_pc)
            tf.summary.scalar('recons_pc_alpha', diff_pc*recons_loss_alpha)
            tf.summary.scalar('recons_edge', diff_edge)
            tf.summary.scalar('recons_edge_alpha', diff_edge_alpha*diff_edge)
            tf.summary.scalar('recons_edge_from_pc', diff_edge_from_pc)
            tf.summary.scalar('recons_edge_from_pc_alpha', diff_edge_from_pc*diff_edge_from_pc_alpha)


            print "--- Get training operator"
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            # loss map 1 edge
            #total_loss = diff_edge_alpha*diff_edge +diff_edge_from_pc*diff_edge_from_pc_alpha
            # loss map 1 pc
            #total_loss = diff_edge_alpha*diff_edge +diff_pc*recons_loss_alpha
            # loss map 1 pc + edge
            total_loss = diff_edge_alpha*diff_edge +diff_pc*recons_loss_alpha +diff_edge_from_pc*diff_edge_from_pc_alpha


            tf.summary.scalar("total_loss",total_loss)
            pc_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='reconstruction_pc')
            edge_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='edge_ae')

            saver_pretrain_edge= tf.train.Saver(var_list = edge_var)
            saver_pretrain_pc= tf.train.Saver(var_list = pc_var)

            update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
            trans_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='to_pc')+ tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='to_edge')
            with tf.control_dependencies(update_ops):
                init_op = optimizer.minimize(total_loss, global_step=batch,var_list = trans_var)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()


        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        saver_pretrain_edge.restore(sess, MODEL_PATH)
        saver_pretrain_pc.restore(sess, MODEL_PATH2)
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'init_op': init_op,
               'total_loss':total_loss,
               'merged': merged,
               'step': batch,
               'end_points': end_points
               }

        best_loss = 1e20
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            train_one_epoch(sess, ops, train_writer,ops['init_op'])

            epoch_loss = eval_one_epoch(sess, ops, test_writer)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model_epoch_%03d.ckpt"%(epoch)))
                log_string("Model saved in file: %s" % save_path)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 3))
    batch_data_shuffled = np.zeros((bsize, NUM_POINT, 3))
    for i in range(bsize):
        ps_shuffled,ps = dataset[idxs[i+start_idx]]
        batch_data[i,...] = ps
        batch_data_shuffled[i,...] = ps_shuffled
    return batch_data_shuffled,batch_data

def train_one_epoch(sess, ops, train_writer, train_op):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)/BATCH_SIZE
    log_string(str(datetime.now()))

    loss_sum = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data_shuffled,batch_data = get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx)
        # Augment batched point clouds by rotation
        if FLAGS.no_rotation:
            aug_data = batch_data_shuffled
            aug_data_unshuffled = batch_data
        else:
            aug_data, aug_data_unshuffled = part_dataset.rotate_point_cloud(batch_data_shuffled, batch_data )

        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['labels_pl']: aug_data_unshuffled,
                     ops['is_training_pl']: is_training,
                     }
        summary, step, _,  total_loss_val, = sess.run([ops['merged'], ops['step'],train_op,  ops['total_loss']], feed_dict=feed_dict)

        train_writer.add_summary(summary, step)

        loss_sum+=total_loss_val

        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            loss_sum = 0


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    num_batches = len(TEST_DATASET)/BATCH_SIZE

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    loss_sum = 0
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data_shuffled, batch_data = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)

        feed_dict = {ops['pointclouds_pl']: batch_data_shuffled,
                     ops['labels_pl']: batch_data,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val = sess.run([ops['merged'], ops['step'],
            ops['total_loss']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        #print('iso loss val  {}'.format(iso_loss_val))
        loss_sum += loss_val
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))

    EPOCH_CNT += 1
    return loss_sum/float(num_batches)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
