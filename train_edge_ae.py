import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
from plyfile import PlyData
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR) # model
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'data_prep'))
import part_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=2, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model_edge_ae', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log_edge_ae', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1000, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=3500, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.000001, help='Initial learning rate [default: 0.001]')#added one 0
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=466*32*200, help='Decay step for lr decay [default: 200000]')# 1000000
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--no_rotation', default = False,action='store_true', help='Disable random rotation during training.')
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

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.1
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP =  466*32*200
BN_DECAY_CLIP = 0.99
HOSTNAME = socket.gethostname()

DATA_PATH = "data/sliced"
EDGES_PATH = "data/template.ply"

TRAIN_DATASET = part_dataset.Dataset(root=DATA_PATH, npoints=NUM_POINT,  split='train', edges=EDGES_PATH)
TEST_DATASET = part_dataset.Dataset(root=DATA_PATH, npoints=NUM_POINT,  split='val',edges=EDGES_PATH)

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
    learning_rate = tf.maximum(learning_rate_base, 0.0000001) # CLIP THE LEARNING RATE!
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

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)
            print("--- Get model and loss")
            # Get model and loss

            # get mesh edges
            plydata = PlyData.read(EDGES_PATH)
            faces = plydata['face']
            faces = np.array([faces[i][0] for i in range(faces.count)])
            with tf.variable_scope("edge_ae"):
                label_edge_lengths = MODEL.compute_edge_lengths(labels_pl, faces=faces)
                pred_points, end_points = MODEL.get_model(label_edge_lengths, is_training_pl, bn_decay=bn_decay)
            num_edges =label_edge_lengths.shape[1]
            recons_edge =  tf.reduce_mean(tf.square(label_edge_lengths - pred_points))

            pairs = MODEL.get_pairs(BATCH_SIZE)
            pairs = tf.random.shuffle(pairs)[:32]
            latent_pairs = tf.gather(end_points['embedding'], pairs)
            recons_pairs = tf.gather(pred_points, pairs)
            interpolated_l = (latent_pairs[:,0] + latent_pairs[:,1]) / 2.0
            interpolated_recons = (recons_pairs[:,0] + recons_pairs[:,1]) / 2.0

            with tf.variable_scope("edge_ae"):
                interpolated_pred = MODEL.get_decoder(interpolated_l, is_training_pl, bn_decay, 2994, BATCH_SIZE)
            edge_lin_loss = tf.reduce_mean(tf.square(interpolated_recons - interpolated_pred))

            edge_recons_loss_alpha =100.0
            edge_lin_loss_alpha = 100.0

            tf.summary.scalar('edge_recons_loss', recons_edge*edge_recons_loss_alpha)
            tf.summary.scalar('edge_lin_loss', edge_lin_loss*edge_lin_loss_alpha)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            total_loss = recons_edge*edge_recons_loss_alpha + edge_lin_loss*edge_lin_loss_alpha

            tf.summary.scalar("total_loss",total_loss)
            update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                init_op = optimizer.minimize(total_loss, global_step=batch)

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
    is_training = True

    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)//BATCH_SIZE
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
    num_batches = len(TEST_DATASET)//BATCH_SIZE

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
        loss_sum += loss_val
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))

    EPOCH_CNT += 1
    return loss_sum/float(num_batches)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
