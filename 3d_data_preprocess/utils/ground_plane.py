import numpy as np

def ransac(X: np.ndarray, y: np.ndarray, iters, tol, seed=12):

    np.random.seed(seed)
    
    N = X.shape[0]
    y = y.reshape(-1, 1)

    sampled_indices = np.random.choice(N, size=(3), replace=False)
    X_sample = X[sampled_indices]
    y_sample = y[sampled_indices]
    best_coeff = (np.linalg.inv(X_sample.T@X_sample)@X_sample.T@y_sample).reshape(-1,1)
    # inliers = np.linalg.norm(X @ best_coeff - y, axis=1) < tol
    inliers = np.abs(X @ best_coeff - y) / np.sqrt(best_coeff[0]**2 + best_coeff[1]**2 + 1) < tol
    

    for i in range(1, iters):
        sampled_indices = np.random.choice(N, size=(4), replace=False)
        X_sample = X[sampled_indices]
        y_sample = y[sampled_indices]
        if np.isclose(np.linalg.det(X_sample.T @ X_sample), 0):
            continue
        cur_coeff = (np.linalg.inv(X_sample.T@X_sample)@X_sample.T@y_sample).reshape(-1,1)
        # cur_inliers = np.linalg.norm(X @ cur_coeff - y, axis=1) < tol
        cur_inliers = np.abs(X @ cur_coeff - y) / np.sqrt(cur_coeff[0]**2 + cur_coeff[1]**2 + 1) < tol
        if np.sum(cur_inliers) > np.sum(inliers):
            best_coeff = cur_coeff
            inliers = cur_inliers

    inliers = inliers.ravel()
    X_inliers = X[inliers]
    y_inliers = y[inliers]
    best_coeff = np.linalg.inv(X_inliers.T@X_inliers)@X_inliers.T@y_inliers


    return best_coeff.flatten(), inliers.astype(bool)


def ransac_1d(y: np.ndarray, tol, iters=1000, seed=12):

    np.random.seed(seed)
    
    N = y.shape[0]
    y = y.reshape(-1, 1)

    sampled_index = np.random.choice(N, size=(1), replace=False)
    y_sample = y[sampled_index]
    inliers = np.abs(y_sample - y) < tol
    

    for i in range(1, iters):
        sampled_index = np.random.choice(N, size=(1), replace=False)
        y_sample = y[sampled_index]
        cur_inliers = np.abs(y_sample - y) < tol
        if np.sum(cur_inliers) > np.sum(inliers):
            best_coeff = y_sample
            inliers = cur_inliers

    y_inliers = y[inliers]
    best_coeff = np.mean(y_inliers)

    return best_coeff.flatten(), inliers.astype(bool)
