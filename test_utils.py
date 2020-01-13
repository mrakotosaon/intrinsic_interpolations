import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import MeshProcess
import numpy as np
import trimesh
from plyfile import PlyData


def get_latent_pc(sess, ops, pc):
    feed_dict = {ops['labels_pl']: pc}
    return sess.run(ops['latent_repr_pc'], feed_dict=feed_dict)

def get_latent_pc_proj(sess, ops, pc):
    # in edge space
    feed_dict = {ops['labels_pl']: pc}
    return sess.run(ops['from_pc_latent'], feed_dict=feed_dict)

def get_latent_edge(sess, ops, pc):
    feed_dict = {ops['labels_pl']: pc}
    return sess.run(ops['latent_repr_edge'], feed_dict=feed_dict)


def decode_pc(sess, ops, repr):
    feed_dict = {ops['latent_pc_pl']: repr}
    return sess.run([ops['pc_decoded'], ops['edge_pc_decoded']], feed_dict=feed_dict)

def get_volume_iso(sess, ops, repr):
    feed_dict = {ops['labels_pl']: repr}
    return sess.run(ops['volume'], feed_dict=feed_dict)



def decode_pc_proj(sess, ops, repr):
    # in edge space
    feed_dict = {ops['latent_proj_pl']: repr}
    return sess.run([ops['pc_decoded_proj'], ops['edge_pc_decoded_proj']], feed_dict=feed_dict)

def decode_edge(sess, ops, repr):
    feed_dict = {ops['latent_edge_pl']: repr}
    return sess.run(ops['edge_decoded'], feed_dict=feed_dict)

def reconstruct_proj(sess, ops, pc):
    feed_dict = {ops['labels_pl']: pc}
    return sess.run([ops['pc_recons_proj'],ops['edge_pc_proj']], feed_dict=feed_dict)

def reconstruct_normal(sess, ops, pc):
    feed_dict = {ops['labels_pl']: pc}
    return sess.run([ops['pc_recons_normal'],ops['edge_pc_normal']], feed_dict=feed_dict)

def reconstruct_ae_pred(sess, ops, pc):
    feed_dict = {ops['labels_pl']: pc}
    return sess.run(ops['edge_ae_pred'], feed_dict=feed_dict)

def reconstruct_proj_pcloss(sess, ops, pc):
    feed_dict = {ops['labels_pl']: pc}
    return sess.run(ops['icp_pc_proj'], feed_dict=feed_dict)

def reconstruct_normal_pcloss(sess, ops, pc):
    feed_dict = {ops['labels_pl']: pc}
    return sess.run(ops['icp_pc_normal'], feed_dict=feed_dict)

def reconstruct_proj_pcloss_frompred(sess, ops, pc, label):
    feed_dict = {ops['labels_pl']: label, ops['pc_recons_proj']: pc}
    return sess.run(ops['icp_pc_proj'], feed_dict=feed_dict)

def reconstruct_normal_pcloss_frompred(sess, ops, pc, label):
    feed_dict = {ops['labels_pl']: label, ops['pc_recons_normal']: pc}
    return sess.run(ops['icp_pc_normal'], feed_dict=feed_dict)


def pc_normalize_area(pc,triv):
    area = MeshProcess.compute_real_mesh_surface_area(pc, triv)
    alpha = 1.66/area
    #alpha = 0.5/area

    pc = pc*np.sqrt(alpha)
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    return pc


def eval_edge_length_variance_interpolation(interpolated_shapes, n_interp = 10):
    return np.sum(np.square(interpolated_shapes[:-1] - interpolated_shapes[1:]))

def eval_area_variance_interpolation(interpolated_shapes,faces, n_interp = 10):
    areas = np.array([compute_area_faces(x, faces) for x in interpolated_shapes])
    return np.sum(np.square(areas[:-1] - areas[1:]))

def compute_area(points, faces):
    mesh = trimesh.Trimesh(vertices=points,
                        faces=faces,
                        process=False)
    return mesh.area_faces.sum()

def compute_area_faces(points,faces):
    mesh = trimesh.Trimesh(vertices=points,
                        faces=faces,
                        process=False)
    return mesh.area_faces


def build_edge_path(sess, ops,couple, decoded_path, test_dataset):
    edge_path_tmp = np.array([x[1][0] for x in decoded_path])
    gt_edge_s = sess.run(ops['gt_edge'], feed_dict={ops['labels_pl']: [test_dataset[couple[0]][1]]})
    gt_edge_t = sess.run(ops['gt_edge'], feed_dict={ops['labels_pl']: [test_dataset[couple[1]][1]]})
    edge_path_tmp[0] = gt_edge_s[0]
    edge_path_tmp[-1] = gt_edge_t[0]
    return edge_path_tmp


def build_edge_path0(sess, ops,couple, decoded_path, test_dataset):
    edge_path_tmp = np.array([x[0] for x in decoded_path])
    gt_edge_s = sess.run(ops['gt_edge'], feed_dict={ops['labels_pl']: [test_dataset[couple[0]][1]]})
    gt_edge_t = sess.run(ops['gt_edge'], feed_dict={ops['labels_pl']: [test_dataset[couple[1]][1]]})
    edge_path_tmp[0] = gt_edge_s[0]
    edge_path_tmp[-1] = gt_edge_t[0]
    return edge_path_tmp

def build_pc_path(sess, ops,couple, decoded_path, test_dataset,replace=True):
    pc_path_tmp = np.array([x[0][0] for x in decoded_path])
    if replace:
        pc_path_tmp[0] = test_dataset[couple[0]][1]
        pc_path_tmp[-1] = test_dataset[couple[1]][1]
    return pc_path_tmp
