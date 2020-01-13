import numpy as np
from scipy.sparse import csr_matrix, spdiags
import time

def cotangent_laplacian(S, area_type = "barycentric"):
    '''
    Compute the cotangent matrix and the weight matrix of a shape
    :param S: a shape with VERT and TRIV
    :return: cotangent matrix W, and the weight matrix A
    '''
    T1 = S.TRIV[:, 0]
    T2 = S.TRIV[:, 1]
    T3 = S.TRIV[:, 2]

    V1 = S.VERT[T1, :]
    V2 = S.VERT[T2, :]
    V3 = S.VERT[T3, :]

    L1 = np.linalg.norm(V2 - V3, axis=1)
    L2 = np.linalg.norm(V1 - V3, axis=1)
    L3 = np.linalg.norm(V1 - V2, axis=1)
    L = np.column_stack((L1, L2, L3))  # Edges of each triangle
    Cos1 = (L2 ** 2 + L3 ** 2 - L1 ** 2) / (2 * L2 * L3)
    Cos2 = (L1 ** 2 + L3 ** 2 - L2 ** 2) / (2 * L1 * L3)
    Cos3 = (L1 ** 2 + L2 ** 2 - L3 ** 2) / (2 * L1 * L2)
    Cos = np.column_stack((Cos1, Cos2, Cos3))  # Cosines of opposite edges for each triangle
    Ang = np.arccos(Cos)  # Angles
    I = np.concatenate((T1, T2, T3))
    J = np.concatenate((T2, T3, T1))
    w = 0.5 * cotangent(np.concatenate((Ang[:, 2], Ang[:, 0], Ang[:, 1]))).astype(float)
    In = np.concatenate((I, J, I, J))
    Jn = np.concatenate((J, I, I, J))
    wn = np.concatenate((-w, -w, w, w))
    if np.isinf(wn).any() or np.isnan(wn).any() :
        print("################################################# there are nan")
        #wn = np.nan_to_num(wn)
        wn[np.isneginf(wn)] = -999999999.0
        wn[np.isposinf(wn)] = 999999999.0
        wn[np.isnan(wn)] = 0.0

    W = csr_matrix((wn, (In, Jn)), [S.nv, S.nv])  # Sparse Cotangent Weight Matrix
    cA = cotangent(Ang) / 2  # Half cotangent of all angles
    At = 1 / 4 * (L[:, [1, 2, 0]] ** 2 * cA[:, [1, 2, 0]] + L[:, [2, 0, 1]] ** 2 * cA[:, [2, 0, 1]]).astype(
        float)  # Voronoi Area
    if area_type == "voronoi":
        # TODO (to update): the MATLAB version gives the area of the parallelogram, not the triangle
        #Ar = 0.5*np.linalg.norm(np.cross(V1 - V2, V1 - V3), axis=1)  # todo - correct version
        Ar = np.linalg.norm(np.cross(V1 - V2, V1 - V3), axis=1)  # todo - Matlab version


        # Use Ar is ever cot is negative instead of At
        locs = cA[:, 0] < 0
        At[locs, 0] = Ar[locs] / 4
        At[locs, 1] = Ar[locs] / 8
        At[locs, 2] = Ar[locs] / 8

        locs = cA[:, 1] < 0
        At[locs, 0] = Ar[locs] / 8
        At[locs, 1] = Ar[locs] / 4
        At[locs, 2] = Ar[locs] / 8

        locs = cA[:, 2] < 0
        At[locs, 0] = Ar[locs] / 8
        At[locs, 1] = Ar[locs] / 8
        At[locs, 2] = Ar[locs] / 4

        Jn = np.zeros(I.shape[0])
        An = np.concatenate((At[:, 0], At[:, 1], At[:, 2]))
        Area = csr_matrix((An, (I, Jn)), [S.nv, 1])  # Sparse Vector of Area Weights
        In = np.arange(S.nv)
        A = csr_matrix((np.squeeze(np.array(Area.todense())), (In, In)),
                       [S.nv, S.nv])  # Sparse Matrix of Area Weights
    elif area_type == "barycentric":
        Ar = 0.5*np.linalg.norm(np.cross(V1 - V2, V1 - V3), axis = 1)
        T = np.zeros([S.VERT.shape[0], V1.shape[0]])
        for t in [T1, T2, T3]:
            T[t, list(range(V1.shape[0]))] = 1.0 # adjacency matrix T[point_i, triangle_i] = 1 if point_i is part of triangle_i
        Ar= T.dot(Ar)/3.0
        Jn = np.zeros(S.nv)
        In = np.arange(S.nv)
        A = csr_matrix((Ar, (In,Jn)), [S.nv, 1])
        A = csr_matrix((np.squeeze(np.array(A.todense())), (In, In)),[S.nv, S.nv])
    else:
        print('unknown area type')
    return W, A


def cotangent(p):
    return np.cos(p)/np.sin(p)


def compute_vertex_and_face_normals(S):
    '''
    Compute the per-face and per-vertex mesh normals
    :param S: the input triangle mesh
    :return: the per-face normals Nf, and the per-vertex normals Nv
    '''
    F = S.TRIV
    V = S.VERT
    num_face = F.shape[0]
    num_vtx = V.shape[0]
    # per-face normal vector
    Nf = np.cross(V[F[:, 1], :] - V[F[:, 0], :], V[F[:, 2], :] - V[F[:, 1], :])
    Fa = 0.5 * np.sqrt(np.power(Nf, 2).sum(axis=1))

    # normalize each vector to unit length
    for i in range(num_face):
        Nf[i, :] = np.divide(Nf[i, :], np.sqrt(np.sum(np.power(Nf[i, :], 2))))

    # per-vertex normal vector
    Nv = np.zeros(V.shape)
    for i in range(num_face):
        for j in range(3):
            if j == 0:
                la = np.sum(np.power(V[F[i, 0], :] - V[F[i, 1], :], 2))
                lb = np.sum(np.power(V[F[i, 0], :] - V[F[i, 2], :], 2))
                W = Fa[i] / (la * lb)
            elif j == 1:
                la = np.sum(np.power(V[F[i, 1], :] - V[F[i, 0], :], 2))
                lb = np.sum(np.power(V[F[i, 1], :] - V[F[i, 2], :], 2))
                W = Fa[i] / (la * lb)
            else:
                la = np.sum(np.power(V[F[i, 2], :] - V[F[i, 0], :], 2))
                lb = np.sum(np.power(V[F[i, 2], :] - V[F[i, 1], :], 2))
                W = Fa[i] / (la * lb)

            if np.isinf(W) or np.isnan(W):
                W = 0

            Nv[F[i, j], :] = Nv[F[i, j], :] + Nf[i, :] * W

    # normalize each vector to unit length
    for i in range(num_vtx):
        Nv[i, :] = np.divide(Nv[i, :], np.sqrt(np.sum(np.power(Nv[i, :], 2))))

    return Nv, Nf


def find_vertex_one_ring_neighbor(S):
    '''
    Find the one-ring neighbor vertex ID for each vertex
    :param S: given mesh
    :return: vtx_neigh[i] gives the neighboring vertex ID list of the i-th vtx
    '''


def find_edge_list(S):
    '''
    Find the edge list and the weight (from the cotangent matrix)
    :param S: given mesh
    :return: edge list Elist, and the corresponding weight EdgeWeight
    '''

# --------------------------------------------------------------------------
#  functions for constructing orientation operators (by Adrien Poulenard)
# --------------------------------------------------------------------------


def compute_real_mesh_surface_area(pc, triv):
    '''
    Compute the surface area of a mesh
    :param S: given mesh
    :return: surface area
    '''
    return sum(compute_real_face_area(pc, triv))

def compute_real_face_area(pc, triv):
    '''
    Compute the area for each triangle face
    :param S: given mesh
    :return: per-face area (nf-by-1) vector
    '''
    T = triv
    V = pc

    V1 = V[T[:, 0], :]
    V2 = V[T[:, 1], :]
    V3 = V[T[:, 2], :]

    Ar = 0.5 * np.linalg.norm(np.cross(V1 - V2, V1 - V3), axis=1) # todo - correct version
    return Ar

def compute_face_area(S):
    '''
    Compute the area for each triangle face
    :param S: given mesh
    :return: per-face area (nf-by-1) vector
    '''
    T = S.TRIV
    V = S.VERT

    V1 = V[T[:, 0], :]
    V2 = V[T[:, 1], :]
    V3 = V[T[:, 2], :]

    # TODO: (to update) the MATLAB version is wrong
    # Ar = 0.5 * np.linalg.norm(np.cross(V1 - V2, V1 - V3), axis=1) # todo - correct version
    Ar = 0.5 * np.sum(np.power(np.cross(V1 - V2, V1 - V3), 2), axis=1)  # todo - MATLAB version
    return Ar

def compute_mesh_surface_area(S):
    '''
    Compute the surface area of a mesh
    :param S: given mesh
    :return: surface area
    '''
    return sum(compute_face_area(S))


def mass_matrix(S):
    '''
    Compute the mass matrix (the construction is slightly different from that in cotangent_laplacian)
    This can be replaced by simply using S.A
    :param S: given mesh
    :return: return the mass matrix (sparse)
    '''

    T = S.TRIV
    V = S.VERT

    Ar = compute_face_area(S)

    T1 = S.TRIV[:, 0]
    T2 = S.TRIV[:, 1]
    T3 = S.TRIV[:, 2]

    I = np.concatenate((T1, T2, T3))
    J = np.concatenate((T2, T3, T1))
    Mij = (1/12) * np.concatenate((Ar, Ar, Ar))
    Mji = Mij
    Mii = (1/6) * np.concatenate((Ar, Ar, Ar))
    In = np.concatenate((I, J, I))
    Jn = np.concatenate((J, I, I))
    Mn = np.concatenate((Mij, Mji, Mii))

    M = csr_matrix((Mn, (In, Jn)), [S.nv, S.nv])  # Sparse Cotangent Weight Matrix
    return M


def compute_function_grad_on_faces(S, f):
    '''
    Compute the gradient of a function on the faces
    :param S: given mesh
    :param f: a function defined on the vertices
    :return: the gradient of f on the faces of S
    '''
    if S.normals_face is None:
        S.compute_vertex_and_face_normals()

    Nf = S.normals_face
    T = S.TRIV
    V = S.VERT
    V1 = V[T[:, 0], :]
    V2 = V[T[:, 1], :]
    V3 = V[T[:, 2], :]

    Ar = compute_face_area(S)
    Ar = np.tile(np.array(Ar, ndmin=2).T, (1, 3))

    f = np.array(f, ndmin=2).T  # make sure the function f is a column vector
    grad_f = np.divide(np.multiply(np.tile(f[T[:, 0]], (1, 3)), np.cross(Nf, V3 - V2)) +
                       np.multiply(np.tile(f[T[:, 1]], (1, 3)), np.cross(Nf, V1 - V3)) +
                       np.multiply(np.tile(f[T[:, 2]], (1, 3)), np.cross(Nf, V2 - V1))
                       , 2 * Ar)
    return grad_f


def vector_field_to_operator(S, Vf):
    '''
    Convert a vector field to an operator
    :param S: given mesh
    :param Vf: a given vector field
    :return: an operator "equivalent" to the vector field
    '''
    if S.normals_face is None:
        S.compute_vertex_and_face_normals()

    Nf = S.normals_face
    T = S.TRIV
    V = S.VERT
    V1 = V[T[:, 0], :]
    V2 = V[T[:, 1], :]
    V3 = V[T[:, 2], :]

    Jc1 = np.cross(Nf, V3 - V2)
    Jc2 = np.cross(Nf, V1 - V3)
    Jc3 = np.cross(Nf, V2 - V1)

    T1 = S.TRIV[:, 0]
    T2 = S.TRIV[:, 1]
    T3 = S.TRIV[:, 2]
    I = np.concatenate((T1, T2, T3))
    J = np.concatenate((T2, T3, T1))

    Sij = 1 / 6 * np.concatenate((np.sum(np.multiply(Jc2, Vf), axis=1),
                                  np.sum(np.multiply(Jc3, Vf), axis=1),
                                  np.sum(np.multiply(Jc1, Vf), axis=1)))

    Sji = 1 / 6 * np.concatenate((np.sum(np.multiply(Jc1, Vf), axis=1),
                                  np.sum(np.multiply(Jc2, Vf), axis=1),
                                  np.sum(np.multiply(Jc3, Vf), axis=1)))

    In = np.concatenate((I, J, I, J))
    Jn = np.concatenate((J, I, I, J))
    Sn = np.concatenate((Sij, Sji, -Sij, -Sji))
    W = csr_matrix((Sn, (In, Jn)), [S.nv, S.nv])
    M = mass_matrix(S)
    tmp = spdiags(np.divide(1, np.sum(M, axis=1)).T, 0, S.nv, S.nv)

    op = tmp.dot(W)
    return op


def compute_orientation_operator_from_a_descriptor(S, B, f):
    '''
    Extract the orientation information from a given descriptor
    :param S: given mesh
    :param B: the basis (should be consistent with the fMap)
    :param f: a descriptor defined on the mesh
    :return: the orientation operator (preserved via commutativity by a fMap)
    '''
    if S.normals_face is None:
        S.compute_vertex_and_face_normals()

    Nf = S.normals_face

    # normalize the gradient to unit length
    grad_f = compute_function_grad_on_faces(S, f)
    length = np.sqrt(np.sum(np.power(grad_f, 2), axis=1)) + 1e-16
    tmp = np.tile(np.array(length, ndmin=2).T, (1, 3))
    norm_grad = np.divide(grad_f, tmp)

    # rotate the gradient by pi/2
    rot_norm_grad = np.cross(Nf, norm_grad)

    # TODO: check which operator should be used
    # Op = vector_field_to_operator(S, norm_grad)
    # diff_Op = np.matmul(B.transpose(), np.matmul(S.A.toarray(), np.matmul(Op, B)))

    # convert vector field to operators
    Op_rot = vector_field_to_operator(S, rot_norm_grad)

    # create 1st order differential operators associated with the vector fields
    diff_Op_rot = np.matmul(B.transpose(), np.matmul(S.A.toarray(), Op_rot.dot(B)))

    return diff_Op_rot

# --------------------------------------------------------------------------
#  functions for constructing orientation operators  - End
# --------------------------------------------------------------------------
