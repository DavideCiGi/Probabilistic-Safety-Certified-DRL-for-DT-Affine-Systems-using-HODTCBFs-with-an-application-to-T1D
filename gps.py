import gpflow
import itertools
import tensorflow as tf

from gpflow.kernels import RBF
from gpflow.kernels import Constant
from gpflow.kernels import Linear

from gpflow.utilities.model_utils import add_likelihood_noise_cov


def beta(j, k, gamma):
    total = 0
    # 1 <= j <= k
    for comb in itertools.combinations(range(k), k-j):
        p = 1.0
        for i in comb:
            p *= (gamma[i]-1.0)
        total += p
    return total


def GP_Delta_kernel(state_dim, action_dim, r):

    kernel = None
    sigma = gpflow.Parameter(1.0, trainable=False)
    for j in range(1, r + 1):
        if kernel is None:
            kernel = (Linear(variance=sigma, active_dims=list(range(state_dim + (j - 1), state_dim + j))) *
                              RBF(lengthscales=[1.0] * state_dim, active_dims=list(range(state_dim))))
        else:
            kernel += (Linear(variance=sigma, active_dims=list(range(state_dim + (j - 1), state_dim + j))) *
                               RBF(lengthscales=[1.0] * state_dim, active_dims=list(range(state_dim))))

    for k in range(1, action_dim + 1):
        kernel += (Linear(variance=sigma, active_dims=list(range(state_dim + r + (k - 1),
                                                                         state_dim + r + k))) *
                           RBF(lengthscales=[1.0] * state_dim, active_dims=list(range(state_dim))))
    return kernel


def preliminary_computations_for_mnsrc(model, action_dim):
    X_train = model.data[0]
    Y_train = model.data[1]
    r = len(model.kernel.kernels) - action_dim
    state_dim = len(model.kernel.kernels[0].kernels[1].active_dims)

    beta_rows = tf.concat([tf.linalg.matrix_transpose(X_train[:, state_dim:state_dim + r]),
                           tf.linalg.matrix_transpose(X_train[:, state_dim + r:])], axis=0)
    # print(f'beta rows: {beta_rows.numpy()}')

    ks = add_likelihood_noise_cov(model.kernel(X_train), model.likelihood, X_train)
    L_hat = tf.linalg.cholesky(ks)

    # eigs = tf.linalg.eigvalsh(ks).numpy()
    # print('Minimum eigenvalue for ks:', eigs.min())

    temp = tf.linalg.solve(L_hat, Y_train)
    m_right_factor = tf.linalg.solve(tf.linalg.matrix_transpose(L_hat), temp)
    return beta_rows, m_right_factor, L_hat


def compute_mean_and_square_root_covariance(x, model, beta_rows, m_right_factor, L_hat):  # pare corretto
    X_train = model.data[0]
    r = len(model.kernel.kernels) - 1  # - action_dim instead of -1 should be correct.

    lambda_row_list = []
    lambda_diag_list = []
    for j in range(r+1):
        lambda_row_list.append(model.kernel.kernels[j].kernels[1](x, X_train))  # I removed kernels[1].K(x, X_train)
        lambda_diag_list.append(tf.squeeze(model.kernel.kernels[j].kernels[1](x, x)))  # I removed kernels[1].K(x, x)
    lambda_rows = tf.concat(lambda_row_list, axis=0)
    k_bar = lambda_rows * beta_rows
    lambda_diag_seq = tf.stack(lambda_diag_list, axis=0)
    lambda_diag = tf.linalg.diag(lambda_diag_seq)
    m = k_bar @ m_right_factor
    Sigma_factor_matrix = tf.linalg.solve(L_hat, tf.linalg.matrix_transpose(k_bar))
    Sigma = lambda_diag - tf.linalg.matrix_transpose(Sigma_factor_matrix) @ Sigma_factor_matrix

    # Sigma_eigs = tf.linalg.eigvalsh(Sigma)
    # Sigma_min_eig = tf.reduce_min(Sigma_eigs)
    jitter = 1e-6
    jitter_eye = jitter * tf.eye(tf.shape(Sigma)[0], dtype=Sigma.dtype)
    Sigma_jittered = Sigma + jitter_eye

    # Sigma_jittered = tf.cond(
    #     tf.less_equal(Sigma_min_eig, 0.0),
    #     lambda: Sigma + jitter_eye,
    #     lambda: Sigma
    # )

    L = tf.linalg.cholesky(Sigma_jittered)
    L_bar = tf.transpose(L)
    Lr_bar, L1_bar = tf.split(L_bar, [r, 1], axis=1)
    m_r, m_1 = tf.split(m, [r, 1], axis=0)
    return m_r, m_1, Lr_bar, L1_bar


