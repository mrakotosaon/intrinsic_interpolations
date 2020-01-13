import tensorflow as tf
import numpy as np
import argparse
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR) # model
sys.path.append(ROOT_DIR) # provider
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import part_dataset
import tf_util
from plyfile import PlyData
import MeshProcess
import trimesh
import test_utils

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=1000, help='Point Number [default: 2048]')
parser.add_argument('--model_path', type=str, default="log_mapping/best_model_epoch_656.ckpt", help='path to the model to evaluate')

FLAGS = parser.parse_args()

NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu

DATA_PATH = "data/sliced"
EDGES_PATH = "data/template.ply"

TEST_DATASET = part_dataset.Dataset(root=DATA_PATH, npoints=NUM_POINT, split='test_all')

MODEL = importlib.import_module("model_edge_ae") # import network module
MODEL2 = importlib.import_module("model_shape_vae") # import network module

MODEL_PATH = FLAGS.model_path

plydata = PlyData.read(EDGES_PATH)
FACES = plydata['face']
FACES = np.array([FACES[i][0] for i in range(FACES.count)])
def get_model(batch_size, num_point):
    with tf.Graph().as_default() as g:
        with tf.device('/gpu:'+str(GPU_INDEX)):
            n_latent_pc = 256
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(1, num_point)
            is_training_false = tf.constant(False)
            latent_pc_pl = tf.placeholder(tf.float32, shape=(1, n_latent_pc))
            latent_edge_pl = tf.placeholder(tf.float32, shape=(1, 256))
            latent_proj_pl = tf.placeholder(tf.float32, shape=(1, 256))
            def to_pc_translator(net):
                net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training_false, scope='to_pc_1', bn_decay=None)
                net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training_false, scope='to_pc_2', bn_decay=None)
                net = tf_util.fully_connected(net, n_latent_pc, activation_fn=None, scope='to_pc_3')
                return net

            def to_edge_translator(net):
                net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training_false, scope='to_edge_1', bn_decay=None)
                net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training_false, scope='to_edge_2', bn_decay=None)
                net = tf_util.fully_connected(net, 256, activation_fn=None, scope='to_edge_3')
                return net
            # proj ae
            end_points = {}
            plydata = PlyData.read(EDGES_PATH)
            faces = plydata['face']
            faces = np.array([faces[i][0] for i in range(faces.count)])
            with tf.variable_scope("edge_ae"):
                label_edge_lengths = MODEL.compute_edge_lengths(labels_pl, faces)
                latent_repr_edge = MODEL.get_fc_encoder(tf.expand_dims(label_edge_lengths, -1),  is_training_false, None, label_edge_lengths.shape[1],batch_size, end_points)
                edge_pred = MODEL.get_decoder(latent_repr_edge, is_training_false, None, label_edge_lengths.shape[1], batch_size)
            with tf.variable_scope("reconstruction_pc"):
                latent_repr_pc = MODEL2.get_encoder_vae(tf.expand_dims(labels_pl, -1), 3, is_training_false, None, num_point,batch_size, end_points)
            with tf.variable_scope("to_edge"):
                from_pc_latent = to_edge_translator(latent_repr_pc)
            with tf.variable_scope("to_pc"):
                from_edge_latent = to_pc_translator(from_pc_latent)
            with tf.variable_scope("reconstruction_pc"):
                end_points = {}
                pc_recons_proj = MODEL2.get_decoder(from_edge_latent, is_training_false, None, num_point, batch_size)
                edge_pc_proj = MODEL.compute_edge_lengths(pc_recons_proj, faces)
                pc_recons_normal = MODEL2.get_decoder(latent_repr_pc, is_training_false, None, num_point, batch_size)
                edge_pc_normal = MODEL.compute_edge_lengths(pc_recons_normal, faces)
                # get icp loss
                icp_pc_proj = tf.reduce_mean(tf.square(pc_recons_proj - MODEL2.ICP(labels_pl, pc_recons_proj, num_point)))
                rotated=  MODEL2.ICP( pc_recons_proj,labels_pl, num_point)
                rotation_angle = np.pi
                cosval = np.cos(rotation_angle)
                sinval = np.sin(rotation_angle)
                rotation_matrix = tf.constant(np.array([[cosval, 0, sinval],
                                            [0, 1, 0],
                                            [-sinval, 0, cosval]]), dtype = tf.float32)
                rotated_pc = tf.expand_dims(tf.transpose(tf.matmul(rotation_matrix,tf.transpose(rotated[0]))), 0)


                icp_pc_normal =  tf.reduce_mean(tf.square(pc_recons_normal - MODEL2.ICP(labels_pl, pc_recons_normal, num_point)))
            gt_edge = MODEL.compute_edge_lengths(labels_pl, faces)

            with tf.variable_scope("edge_ae"):
                edge_decoded = MODEL.get_decoder(latent_edge_pl, is_training_false, None, label_edge_lengths.shape[1], batch_size)
            with tf.variable_scope("to_pc"):
                from_edge_latent_dec = to_pc_translator(latent_proj_pl)
            with tf.variable_scope("reconstruction_pc"):
                pc_decoded = MODEL2.get_decoder(latent_pc_pl, is_training_false, None, num_point, batch_size)
                edge_pc_decoded = MODEL.compute_edge_lengths(pc_decoded, faces)
                pc_decoded_proj = MODEL2.get_decoder(from_edge_latent_dec, is_training_false, None, num_point, batch_size)
                edge_pc_decoded_proj = MODEL.compute_edge_lengths(pc_decoded_proj, faces)
            # align 2 pc
            aligned_pc =  MODEL2.ICP(pointclouds_pl, labels_pl, num_point)
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver_pretrain = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        init = tf.global_variables_initializer()
        sess.run(init)
        saver_pretrain.restore(sess, MODEL_PATH)

        ops = {'labels_pl': labels_pl,
                'pointclouds_pl': pointclouds_pl,
                'gt_edge':gt_edge,
                'aligned_pc': aligned_pc,
               'pc_recons_proj' : pc_recons_proj,
               'pc_recons_normal' : pc_recons_normal,
               'edge_pc_proj': edge_pc_proj,
               'edge_pc_normal': edge_pc_normal,
               'edge_ae_pred': edge_pred,
               'latent_pc_pl':latent_pc_pl,
               'latent_edge_pl':latent_edge_pl,
               'latent_proj_pl':latent_proj_pl,
               'latent_repr_pc':latent_repr_pc,
               'from_pc_latent':from_pc_latent,
               'latent_repr_edge':latent_repr_edge,
               'edge_decoded':edge_decoded,
               'pc_decoded':pc_decoded,
               'pc_decoded_proj':pc_decoded_proj,
               'edge_pc_decoded':edge_pc_decoded,
               'edge_pc_decoded_proj':edge_pc_decoded_proj,
               'icp_pc_proj':icp_pc_proj,
               'icp_pc_normal':icp_pc_normal
               }
        return sess, ops



def eval_interpolations_variance(n_interp=10):
    # build random couples in testset
    np.random.seed(0)
    couples = np.array(range(len(TEST_DATASET)))
    np.random.shuffle(couples)
    couples = np.reshape(couples, [-1, 2])

    sess, ops = get_model(batch_size=1, num_point=NUM_POINT)

    dist_pc_loss = []
    dist_pc_proj_loss = []
    dist_edge_loss = []
    dist_area_loss = []
    dist_area_proj_loss = []
    print("Testing couples of shapes...")
    for j, couple in enumerate(couples):
        if j%50==0:
            print("{}/{}".format(j, len(couples)))
        z_s_pc = test_utils.get_latent_pc(sess, ops,[TEST_DATASET[couple[0]][1]])
        z_s_pc_proj = test_utils.get_latent_pc_proj(sess, ops,[TEST_DATASET[couple[0]][1]])
        z_s_edge = test_utils.get_latent_edge(sess, ops,[TEST_DATASET[couple[0]][1]])

        z_t_pc = test_utils.get_latent_pc(sess, ops,[TEST_DATASET[couple[1]][1]])
        z_t_pc_proj = test_utils.get_latent_pc_proj(sess, ops,[TEST_DATASET[couple[1]][1]])
        z_t_edge = test_utils.get_latent_edge(sess, ops,[TEST_DATASET[couple[1]][1]])

        pc_path =np.array([(1-t)*z_s_pc + (t)*z_t_pc for t in np.linspace(0.0, 1.0, num=n_interp)])
        pc_proj_path =np.array([(1-t)*z_s_pc_proj + (t)*z_t_pc_proj for t in np.linspace(0.0, 1.0, num=n_interp)])
        edge_path =np.array([(1-t)*z_s_edge + (t)*z_t_edge for t in np.linspace(0.0, 1.0, num=n_interp)])

        decoded_path_pc = [test_utils.decode_pc(sess, ops, el) for el in pc_path]
        decoded_path_pc_proj = [test_utils.decode_pc_proj(sess, ops, el) for el in pc_proj_path]
        decoded_path_edge = [test_utils.decode_edge(sess, ops, el) for el in edge_path]

        area_var_proj = test_utils.eval_area_variance_interpolation(test_utils.build_pc_path(sess, ops,couple, decoded_path_pc_proj, TEST_DATASET), FACES)
        area_var_pc = test_utils.eval_area_variance_interpolation(test_utils.build_pc_path(sess, ops,couple, decoded_path_pc, TEST_DATASET), FACES)
        dist_area_loss.append(area_var_pc)
        dist_area_proj_loss.append(area_var_proj)
        edge_path_tmp = test_utils.build_edge_path(sess, ops,couple, decoded_path_pc,TEST_DATASET)
        dist_pc_path = test_utils.eval_edge_length_variance_interpolation(edge_path_tmp)

        edge_path_tmp = test_utils.build_edge_path(sess, ops,couple, decoded_path_pc_proj,TEST_DATASET)
        dist_pc_proj_path = test_utils.eval_edge_length_variance_interpolation(edge_path_tmp)

        edge_path_tmp = test_utils.build_edge_path0(sess, ops,couple, decoded_path_edge,TEST_DATASET)
        dist_edge_path = test_utils.eval_edge_length_variance_interpolation(edge_path_tmp)
        #print("shape_vae: {} ours: {} edge_ae: {}".format(dist_pc_path, dist_pc_proj_path, dist_edge_path))

        dist_pc_loss.append(dist_pc_path)
        dist_pc_proj_loss.append(dist_pc_proj_path)
        dist_edge_loss.append(dist_edge_path)

    print("=========================\nVariance edge length:\nshape_vae: {}\nours:      {}\nedge ae:   {}".format(np.mean(dist_pc_loss), np.mean(dist_pc_proj_loss), np.mean(dist_edge_loss)))
    print("=========================\nVariance area:\nshape_vae: {}\nours:      {}".format(np.mean(dist_area_loss), np.mean(dist_area_proj_loss)))

def align_pc(sess, ops, pc, base):
    feed_dict = {ops['labels_pl']: base, ops['pointclouds_pl'] : pc}
    return sess.run(ops['aligned_pc'], feed_dict=feed_dict)

def interpolate_and_save(A, B, n_interp = 10):
    for i in range(600,650):
        mesh = trimesh.Trimesh(vertices=TEST_DATASET[i][1],faces=FACES,process=False)
        mesh.export("results/test{}.ply".format( i))
    sess, ops = get_model(batch_size=1, num_point=NUM_POINT)
    z_s_pc = test_utils.get_latent_pc(sess, ops,[TEST_DATASET[A][1]])
    z_s_pc_proj = test_utils.get_latent_pc_proj(sess, ops,[TEST_DATASET[A][1]])

    z_t_pc = test_utils.get_latent_pc(sess, ops,[TEST_DATASET[B][1]])
    z_t_pc_proj = test_utils.get_latent_pc_proj(sess, ops,[TEST_DATASET[B][1]])

    pc_path =np.array([(1-t)*z_s_pc + (t)*z_t_pc for t in np.linspace(0.0, 1.0, num=n_interp)])
    pc_proj_path =np.array([(1-t)*z_s_pc_proj + (t)*z_t_pc_proj for t in np.linspace(0.0, 1.0, num=n_interp)])

    decoded_path_pc = [test_utils.decode_pc(sess, ops, el) for el in pc_path]
    decoded_path_pc = test_utils.build_pc_path(sess, ops,[A, B], decoded_path_pc, TEST_DATASET, replace=True) # replace the extremities of interpolation by ground truth

    decoded_path_pc_proj = [test_utils.decode_pc_proj(sess, ops, el) for el in pc_proj_path]
    decoded_path_pc_proj = test_utils.build_pc_path(sess, ops,[A, B], decoded_path_pc_proj, TEST_DATASET, replace=True)


    # align interpolation with A and save
    for i in range(n_interp):
        aligned = align_pc(sess, ops,[decoded_path_pc[i]],[TEST_DATASET[A][1]])[0]
        mesh = trimesh.Trimesh(vertices=aligned,faces=FACES,process=False)
        mesh.export("results/shape_vae_interp_A{}_B{}_n{}.ply".format(A, B, i))
        aligned = align_pc(sess, ops, [decoded_path_pc_proj[i]],[TEST_DATASET[A][1]])[0]
        mesh = trimesh.Trimesh(vertices=aligned,faces=FACES,process=False)
        mesh.export("results/ours_interp_A{}_B{}_n{}.ply".format(A, B, i))


def eval_reconstruction():
    shapes = np.array(range(len(TEST_DATASET)))
    np.random.shuffle(shapes)
    np.random.seed(0)
    sess, ops = get_model(batch_size=1, num_point=NUM_POINT)
    diff_normal = 0
    diff_proj = 0
    diff_edge_ae = 0
    nromal_pcloss = 0
    pc_loss = 0
    edge_losses = []
    edge_losses_proj = []

    area_ae = []
    area_proj = []
    print('testing shapes...')
    for k,shape in enumerate(shapes):
        if k%50==0:
            print("{}/{}".format(k, len(shapes)))
        area_gt = test_utils.compute_area_faces(TEST_DATASET[shape][1], FACES)
        proj_pc, proj_edge = test_utils.reconstruct_proj(sess, ops, [TEST_DATASET[shape][1]])
        area_proj.append(np.mean(np.square(test_utils.compute_area_faces(proj_pc[0], FACES) - area_gt)))



        normal_pc, normal_edge = test_utils.reconstruct_normal(sess, ops, [TEST_DATASET[shape][1]])
        area_ae.append(np.mean(np.square(test_utils.compute_area_faces(normal_pc[0], FACES) - area_gt)))


        gt_edge =  sess.run(ops['gt_edge'], feed_dict={ops['labels_pl']: [TEST_DATASET[shape][1]]})
        ae_edge_pred = test_utils.reconstruct_ae_pred(sess, ops,[TEST_DATASET[shape][1]])
        edge_loss_proj =np.mean(np.square(gt_edge - proj_edge))
        edge_losses_proj.append(edge_loss_proj)
        diff_proj+= edge_loss_proj

        edge_loss_normal = np.mean(np.square(gt_edge - normal_edge))
        edge_losses.append(edge_loss_normal)
        diff_normal +=edge_loss_normal
        diff_edge_ae +=np.mean(np.square(gt_edge - ae_edge_pred))
        pc_loss +=test_utils.reconstruct_proj_pcloss(sess, ops, [TEST_DATASET[shape][1]])
        nromal_pcloss += test_utils.reconstruct_normal_pcloss(sess, ops, [TEST_DATASET[shape][1]])

    print('******** reconstruction losses ********')
    print("edge reconstruction:")
    print("edge_ae:    {}".format(diff_edge_ae/len(shapes)))
    print("ours:       {}".format(diff_proj/len(shapes)))
    print("shape_vae:  {}".format(diff_normal/len(shapes)))
    print("====================")
    print('pc reconstruction:')
    print("ours:       {}".format(pc_loss/len(shapes)))
    print("shape_vae:  {}".format(nromal_pcloss/len(shapes)))
    print("====================")
    print("area recontruction:")
    print("shape_vae:  {}".format(np.mean(np.array(area_ae))))
    print("ours:       {}".format(np.mean(np.array(area_proj))))



if __name__=='__main__':
    #   eval_reconstruction()
    interpolate_and_save(0, 617)
    #eval_interpolations_variance()
