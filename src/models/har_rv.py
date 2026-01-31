import numpy as np

def compute_har_components(rv):
    rv_d = rv.copy()

    rv_w = np.array([
        np.mean(rv[max(0, t-4):t+1]) for t in range(len(rv))
    ])

    rv_m = np.array([
        np.mean(rv[max(0, t-20):t+1]) for t in range(len(rv))
    ])

    return rv_d, rv_w, rv_m


def estimate_har_rv(target, rv_d, rv_w, rv_m):
    X = np.column_stack([np.ones(len(target)-1),
                         rv_d[:-1], rv_w[:-1], rv_m[:-1]])
    y = target[1:]
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta
