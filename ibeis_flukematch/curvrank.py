import ctypes
import numpy as np
from scipy.interpolate import BPoly
from scipy.interpolate import interp1d

costs_lib = ctypes.cdll.LoadLibrary('dtw.so')

ndmat_f_type = np.ctypeslib.ndpointer(
    dtype=np.float32, ndim=2, flags='C_CONTIGUOUS')
ndmat_i_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=2, flags='C_CONTIGUOUS')

dtw_weighted_euclidean_cpp = costs_lib.weighted_euclidean
dtw_weighted_euclidean_cpp.argtypes = [
    ndmat_f_type, ndmat_f_type, ndmat_f_type,
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ndmat_f_type
]


# TODO: find a better way to structure these two functions
def resample(x, length):
    interp = np.linspace(0, length, num=x.shape[0], dtype=np.float32)
    f_interp = interp1d(interp, x, kind='linear')

    resamp = f_interp(np.arange(length))

    return resamp


def resampleNd(X, length):
    Xr = np.zeros((length, X.shape[1]), dtype=np.float32)
    for j in range(X.shape[1]):
        Xr[:, j] = resample(X[:, j], length)

    return Xr


def bernstein_poly(x, coeffs):
    interval = np.array([0, 1])
    f = BPoly(coeffs, interval, extrapolate=False)

    return f(x)


def get_spatial_weights(num_points, coeffs):
    coeffs = coeffs.reshape(coeffs.shape[0], 1)
    weights = bernstein_poly(
        np.linspace(0, 1, num_points), coeffs
    )
    weights = weights.reshape(-1, 1).astype(np.float32)

    return weights


def rotate(radians):
    M = np.eye(3)
    M[0, 0], M[1, 1] = np.cos(radians), np.cos(radians)
    M[0, 1], M[1, 0] = np.sin(radians), -np.sin(radians)

    return M


def reorient(points, theta, center):
    M = rotate(theta)
    points_trans = points - center
    points_aug = np.hstack((points_trans, np.ones((points.shape[0], 1))))
    points_trans = np.dot(M, points_aug.transpose())
    points_trans = points_trans.transpose()[:, :2]
    points_trans += center

    assert points_trans.ndim == 2, 'points_trans.ndim == %d != 2' % (
        points_trans.ndim)

    return points_trans


def oriented_curvature(contour, radii):
    curvature = np.zeros((contour.shape[0], len(radii)), dtype=np.float32)
    # define the radii as a fraction of either the x or y extent
    for i, (x, y) in enumerate(contour):
        center = np.array([x, y])
        dists = ((contour - center) ** 2).sum(axis=1)
        inside = dists[:, np.newaxis] <= (radii * radii)

        for j, _ in enumerate(radii):
            curve = contour[inside[:, j]]

            n = curve[-1] - curve[0]
            theta = np.arctan2(n[1], n[0])

            curve_p = reorient(curve, theta, center)
            center_p = np.squeeze(reorient(center[None], theta, center))
            r0 = center_p - radii[j]
            r1 = center_p + radii[j]
            r0[0] = max(curve_p[:, 0].min(), r0[0])
            r1[0] = min(curve_p[:, 0].max(), r1[0])

            area = np.trapz(curve_p[:, 1] - r0[1], curve_p[:, 0], axis=0)
            curv = area / np.prod(r1 - r0)
            curvature[i, j] = curv

    return curvature


def dtw_weighted_euclidean_star(qcurv_dcurv, weights, window):
    return dtw_weighted_euclidean(*qcurv_dcurv, weights=weights, window=window)


def dtw_weighted_euclidean(qcurv, dcurv, weights, window):
    assert qcurv.dtype == np.float32, 'qcurv.dtype = %s' % qcurv.dtype
    assert dcurv.dtype == np.float32, 'dcurv.dtype = %s' % dcurv.dtype
    assert weights.dtype == np.float32, 'weights.dtype = %s' % weights.dtype
    assert qcurv.flags.c_contiguous
    assert dcurv.flags.c_contiguous
    assert weights.flags.c_contiguous
    assert qcurv.shape == dcurv.shape
    assert qcurv.shape[0] == weights.shape[0]
    assert qcurv.ndim == dcurv.ndim == weights.ndim == 2

    m, n = qcurv.shape
    costs_out = np.full((m, m), np.inf, dtype=np.float32)
    costs_out[0, 0] = 0.
    dtw_weighted_euclidean_cpp(
        qcurv, dcurv, weights, m, n, window,
        costs_out
    )

    return costs_out[-1, -1]
