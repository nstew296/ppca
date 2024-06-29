import numpy as np
from scipy.linalg import orthogonal_procrustes
from ppca_em import ppca_for_incomplete_data
np.random.seed(0)


def test_ppca(n, p, q, p_missing=0.25, missing_pattern='MCAR'):
    sigma_true = 0.7
    mu_true = np.random.uniform(low=-1, high=1, size=p)
    w_true = np.random.uniform(low=-1, high=1, size=(p, q))

    z = np.random.multivariate_normal(np.zeros(q), np.eye(q), size=n)
    noise = sigma_true*np.random.normal(size=(n, p))
    x = (z.dot(w_true.T) + np.broadcast_to(mu_true.T, (n, p))) + noise
    
    if missing_pattern == 'MCAR':
        missing = np.where(p_missing > np.random.rand(n, p))
        x[missing[0], missing[1]] = np.NaN

    elif missing_pattern == 'MNAR':
        missing_prob = 2*p_missing / (1 + np.exp(-1*(x - np.mean(x, axis=0))/np.std(x, axis=0)))
        missing = np.where(missing_prob > np.random.rand(n, p))
        x[missing[0], missing[1]] = np.NaN

    else:
        raise NotImplementedError("Supported missing patterns are MCAR and MNAR.")

    w_out, sigma_out, mu_out = ppca_for_incomplete_data(x, q)
    r = orthogonal_procrustes(w_out, w_true)[0]
    w_out_rotated = w_out.dot(r)

    relative_error_sigma = np.abs(sigma_out - sigma_true) / np.abs(sigma_true)
    relative_error_mu = np.linalg.norm(mu_out - mu_true) / np.linalg.norm(mu_true)
    relative_error_w = np.linalg.norm(w_out_rotated - w_true, ord='fro') / np.linalg.norm(w_true, ord='fro')

    return f"\t Relative error sigma: {relative_error_sigma}, \n" \
           f"\t Relative error mu: {relative_error_mu},  \n" \
           f"\t Relative Error W (Rotated): {relative_error_w} \n"


print("Simulation results with no missing values: \n",
      test_ppca(n=500, p=30, q=3, p_missing=0.0))
print("Simulation results with approximately 10% of values missing completely at random: \n",
      test_ppca(n=500, p=30, q=3, p_missing=0.1, missing_pattern='MCAR'))
print("Simulation results with approximately 10% of values missing not at random: \n",
      test_ppca(n=500, p=30, q=3, p_missing=0.1, missing_pattern='MNAR'))


