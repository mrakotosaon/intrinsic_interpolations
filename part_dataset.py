import os
import os.path
import numpy as np
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import MeshProcess
import trimesh


def pc_normalize(pc):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def pc_normalize_area(pc,triv):
    area = MeshProcess.compute_real_mesh_surface_area(pc, triv)
    alpha = 1.66/area
    pc = pc*np.sqrt(alpha)
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    return pc



def rotate_point_cloud(batch_data_shuffled, batch_data ):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    rotated_data_shuffled = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        shape_pc_shuffled = batch_data_shuffled[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data_shuffled[k, ...] = np.dot(shape_pc_shuffled.reshape((-1, 3)), rotation_matrix)
    return rotated_data_shuffled, rotated_data


class Dataset():
    def __init__(self, root,  npoints = 2500, split='train', normalize=False,  normalize_area=True,edges=None):
        self.normalize = normalize
        self.normalize_area = normalize_area
        if split == "train":
            self.files = np.load(os.path.join(root, "train_train.npy"))
        elif split == "val":
            self.files = np.load(os.path.join(root, "train_val.npy"))
        elif split == "test":
            self.files = np.load(os.path.join(root, "dfaust_test.npy"))
        elif split == "test_all":
            self.files = np.concatenate([np.load(os.path.join(root, "dfaust_test.npy")),np.load(os.path.join(root, "surreal_test.npy"))])

        self.npoints = self.files.shape[1]
        if self.normalize:
            self.files = np.array([pc_normalize(x) for x in self.files])
        if self.normalize_area:
            self.triv = trimesh.load(edges, process=False).faces
            self.files = np.array([pc_normalize_area(x, self.triv) for x in self.files])

    def __getitem__(self, index):
        point_set = self.files[index]
        choice = np.random.choice(len(point_set), self.npoints, replace=True)
        point_set_shuffled = point_set[choice, :]
        return point_set_shuffled, point_set

    def __len__(self):
        return len(self.files)
