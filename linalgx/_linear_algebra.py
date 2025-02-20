import numpy as np

def tounit(vec: np.ndarray) -> np.ndarray:
    """converts into unitvectors , normalized form"""
    if vec.ndim == 1: return vec / np.linalg.norm(vec)
    return vec / np.linalg.norm(vec, axis=1)[:, np.newaxis]


def vec_angle(svec, tvec, normalize=True, maxang=180, signed=False, units="deg") -> np.ndarray:
    """
    Calculates the angles in rad or deg between source vectors to target vector
     Args:
        svec: (3) source vector
        tvec: (N,3) or (3) target vector/s from where to check angle
        normalize: True if normalization is needed to be done on vectors
        maxang: outputs in [0,maxang] range. Usually [0,90] default it is [0,180]
        signed: if clock or anticlockwise angles needed
        units: format of unit required as output, "rad": in radians, "deg": in degrees (default).

    Returns: (N) array of angles between vectors.
    """
    if isinstance(svec, (list, tuple)):
        svec = np.asarray(svec)
    if isinstance(tvec, (list, tuple)):
        tvec = np.asarray(tvec)

    if normalize:
        svec = tounit(svec)
        tvec = tounit(tvec)
    dotprod = np.dot(tvec, svec)
    ang = np.degrees(np.arccos(np.clip(dotprod, -1.0, 1.0)))
    if signed is not None:
        dp = np.dot(tvec, signed)
        k = 1  # code for assigning sign in terms of look at vector. will code later!
    if maxang == 90: ang = np.where(ang > maxang, 180 - ang, ang)
    if units in 'rad': return np.radians(ang).astype(np.float16)
    return ang.astype(np.float16)


def Point2PointDist(points: np.ndarray, ref: np.ndarray, positive=True) -> np.ndarray:
    # Euclidian Distance which is always positive!
    return np.linalg.norm((points - ref), axis=1)


def get_rotmat(vec1, vec2, around_axis = None):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: source vector
    :param vec2: target vector on which the source vector will be rotated
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    vec1 = np.asarray(vec1, dtype=np.float64)
    vec2 = np.asarray(vec2, dtype=np.float64)

    # Normalize input vectors
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)

    v = np.cross(vec1, vec2) if around_axis is None else np.asarray(around_axis , dtype=np.float64)

    if np.allclose(v, 0):
        return np.eye(3)

    vnorm = np.linalg.norm(v)
    c = np.dot(vec1, vec2)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    rotation_mat = np.eye(3) + kmat + np.dot(kmat, kmat) * ((1 - c)/ vnorm ** 2)
    return rotation_mat
