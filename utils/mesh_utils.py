import tensorflow as tf
import numpy as np


def corrected_norm(x):
    return tf.sqrt(tf.math.maximum(tf.reduce_sum(tf.square(x), axis = -1), 1e-8))

def get_area_from_vect(a, b):
    #return 0.5*tf.norm(tf.math.maximum(tf.cross(a,b), 1e-7), axis = 2)
    #return 0.5*tf.norm(tf.cross(a,b), axis = 2)
    return 0.5*corrected_norm(tf.cross(a,b))


def get_area_vector(points, triangles):
    vertex = []
    for i in range(3):
        index_vertex = tf.constant(triangles[:,i], dtype = tf.int32)
        vertex.append(tf.map_fn(lambda x : tf.gather(x,index_vertex), points))
    edges1 = vertex[1] - vertex[2]
    edges2 = vertex[0] - vertex[2]
    edges3 = vertex[0] - vertex[1]
    area_per_triangle = get_area_from_vect(edges3, edges2)
    return area_per_triangle, vertex, [edges1, edges2, edges3]

def init_adjacency_tensor(triangles, n_points):
    adj = np.zeros([n_points, triangles.shape[0]])
    for i,row in enumerate(triangles):
        for point in row:
            adj[int(point)][i] = 1
    return tf.constant(adj, dtype = tf.float32)# tf.float64


def compute_basis(triangles, points, end_points):
    #points =  tf.cast(points, tf.float64)
    area_per_triangle, tri_vertex, tri_edges= get_area_vector(points, triangles)
    A = compute_A(triangles, points, area_per_triangle, end_points)

    A_pow = tf.sqrt(tf.math.maximum(A, 1e-8))
    #A_pow = tf.sqrt(A)

    A_pow_neg = 1.0/A_pow
    A_pow = tf.matrix_diag(A_pow)
    A_pow_neg = tf.matrix_diag(A_pow_neg)
    W = compute_W(triangles, points, tri_edges, end_points)
    W = tf.sparse.reorder(W)
    W = tf.sparse.to_dense(W)
    #L = tf.einsum("bij,bjk->bik", A_inv, W)
    L0 = tf.einsum("bij,bjk,bkl->bil", A_pow_neg, W, A_pow_neg)
    evals, evecs = tf.linalg.eigh(L0)
    evecs= tf.einsum("bij,bjk->bik", A_pow, evecs)
    evals = evals[:, :100]
    evecs =  evecs[:,:, :100]
    return evals, evecs

def compute_A(triangles, points, area_per_triangle, end_points):
    adjacency_matrix = init_adjacency_tensor(triangles, points.shape[1])
    adjacency_matrix = tf.tile(tf.expand_dims(adjacency_matrix, 0), [tf.shape(area_per_triangle)[0],1,1])
    A = tf.einsum("bij,bj->bi", adjacency_matrix, area_per_triangle)/3.0
    # batch_index = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(points.shape[0]),-1), [1, points.shape[1]]), -1)
    # diag_index = tf.expand_dims( tf.tile(tf.expand_dims(tf.range(points.shape[1]), 0), [points.shape[0], 1]), -1)
    # index = tf.concat([batch_index, diag_index, diag_index], -1)
    # index = tf.cast(index, tf.int64)
    # A_inv = tf.SparseTensor(indices=tf.reshape(index, [-1, 3]), values =tf.reshape(A_inv, [-1]), dense_shape=[points.shape[0],points.shape[1], points.shape[1]])
    return A

def cotangent(p):
    return tf.cos(p)/tf.sin(p)

def compute_W(triangles, points, edges, end_points):
    end_points['test'] = []
    edges_length = []
    edges_square = []
    for i in range(3):
        edges_length.append(corrected_norm(edges[i]))

        #edges_length.append(tf.norm(edges[i], axis = -1))
        edges_square.append(tf.square(edges_length[i]))
        #edges_square.append(tf.reduce_sum(tf.square(edges[i]), axis = -1))

    cos0 = (edges_square[1] + edges_square[2] - edges_square[0]) / (2*edges_length[1]*edges_length[2])
    cos1 = (edges_square[2] + edges_square[0] - edges_square[1]) / (2*edges_length[2]*edges_length[0])
    cos2 = (edges_square[0] + edges_square[1] - edges_square[2]) / (2*edges_length[0]*edges_length[1])
    cos = tf.concat([cos2, cos0, cos1], -1)

    end_points['test'].append(points)

    end_points['test'].append(cos)
    angles = tf.math.acos(cos)  # Angles
    #w = tf.reshape(0.5 * cotangent(angles), [cos0.shape[0], cos0.shape[1], 3])
    w = 0.5 * cotangent(angles)
    I =tf.concat([triangles[:,0], triangles[:,1], triangles[:,2]],-1)
    J =tf.concat([triangles[:,1], triangles[:,2], triangles[:,0]], -1)
    In = tf.concat([I,J,I,J], -1)
    Jn = tf.concat([J,I,I,J], -1)
    wn = tf.concat([-w,-w,w,w], -1)
    In = tf.cast(In, tf.int32)
    Jn = tf.cast(Jn, tf.int32)
    linearized_index = tf.matmul(tf.concat([tf.expand_dims(In, -1), tf.expand_dims(Jn, -1)], 1), [[1000],[1]])
    unique_index, idx = tf.unique(tf.squeeze(linearized_index))
    wn_new = tf.map_fn(lambda x : tf.unsorted_segment_sum(x, idx,tf.shape(unique_index)[0]), wn)
    unique_index = tf.expand_dims(unique_index, 1)
    indices = tf.concat([unique_index//1000, unique_index%1000], axis=1)
    indices = tf.tile(tf.expand_dims(indices, 0), [points.shape[0], 1, 1])
    batch_index = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(wn.shape[0]),-1), [1, tf.shape(indices)[1]]), -1)
    batch_index = tf.cast(batch_index, tf.int64)
    indices = tf.cast(indices, tf.int64)
    indices = tf.concat([batch_index,indices], -1)
    wn_new = tf.where(tf.is_inf(wn_new), tf.multiply(tf.math.sign(wn_new), tf.ones_like(wn_new))*1e10, wn_new)
    wn_new = tf.where(tf.is_nan(wn_new), tf.zeros_like(wn_new), wn_new)

    end_points['test'].append(wn_new)
    W = tf.SparseTensor(indices=tf.reshape(indices, [-1, 3]), values =tf.reshape(wn_new, [-1]), dense_shape=[points.shape[0],points.shape[1], points.shape[1]])
    #end_points['test'] = [tf.multiply(tf.math.sign(wn_new), tf.ones_like(wn_new))*999999999.0]
    #end_points['test'].append(tf.sparse.to_dense(W))
    return W
