import numpy as np


def ppca_for_incomplete_data(x, q, tol=1e-3, max_iter=500):
    """
    :param x: Data matrix of dimension nxp
    :param q: Number of latent variables; must be < p
    :param tol: Convergence threshold for EM algorithm (based on Frobenius norm of W)
    :param max_iter: Maximum number of iterations of EM algorithm
    :return: EM estimates for parameters W, sigma, and mu
    """
    n, p = x.shape
    if q >= p:
        raise ValueError("q must be < p")

    # Identify different combinations of observed and unobserved predictors
    # Observations with the same pattern of missingness can be updated together in E step
    missing = np.isnan(x)
    unique_missing_combinations = np.unique(missing, axis=0)

    # Placeholder for z
    z = np.zeros((n, q))

    # Impute missing values with column means
    mu = np.nanmean(x, axis=0)
    x_imputed = x.copy()
    x_imputed[missing] = np.take(mu, np.where(missing)[1])
    sample_covariance_imputed = np.cov(x_imputed.T)

    # Perform eigen-decomposition based the imputed data
    eigenvalues, eigenvectors = np.linalg.eig(sample_covariance_imputed)
    eigenvalue_index_sorted = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, eigenvalue_index_sorted]
    eigenvalues = eigenvalues[eigenvalue_index_sorted]

    # Initialize sigma and W by using the closed-form MLE calculated on the imputed data
    sigma = np.sqrt(np.mean(eigenvalues[q:]))
    w = eigenvectors[:, :q].dot(np.sqrt((eigenvalues[:q] - sigma ** 2) * np.eye(q)))

    count = 0
    while count <= max_iter:
        for missing_indices in unique_missing_combinations:
            # Partition the parameters based on the observed and unobserved indices
            mu_observed = mu[~missing_indices]
            mu_unobserved = mu[missing_indices]
            w_observed = w[~missing_indices, :]
            w_unobserved = w[missing_indices, :]
            m_inv_observed = np.linalg.inv(w_observed.T.dot(w_observed) + (sigma ** 2 * np.eye(q)))

            # Identify which observations share this combination of missing predictors
            combination_group = np.all(missing == missing_indices, axis=1)
            x_observed = x[np.ix_(combination_group, ~missing_indices)]

            # Update the latent variables z for this group of observations (E-Step)
            group_z_latent = (m_inv_observed.dot(w_observed.T).dot((x_observed - mu_observed).T)).T
            z[combination_group, :] = group_z_latent

            # Update the estimate for the missing values
            if np.sum(missing_indices) > 0:
                x_unobserved = w_unobserved.dot(group_z_latent.T) + mu_unobserved.reshape(-1, 1)
                x[np.ix_(combination_group, missing_indices)] = x_unobserved.T

        m_inv = np.linalg.inv(w.T.dot(w) + (sigma ** 2 * np.eye(q)))

        # Updates the parameters to maximize the expected joint log-likelihood (M-Step)
        w_old = w
        w = (x - mu).T.dot(z).dot(np.linalg.inv(n * sigma ** 2 * m_inv + z.T.dot(z)))
        sigma = np.sqrt((1/(n*p)) * (np.trace((x-mu).T.dot(x-mu)) - 2*np.trace(z.dot(w.T).dot((x-mu).T))
                                     + np.trace((n * sigma ** 2 * m_inv + z.T.dot(z)).dot(w.T.dot(w)))))
        mu = np.mean(x - z.dot(w.T), axis=0)

        # Check for convergence based on W, the primary parameter of interest
        change = np.linalg.norm(np.abs(w - w_old), ord='fro')
        if change <= tol:
            break
        count += 1

    return w, sigma, mu
