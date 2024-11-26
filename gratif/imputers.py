import numpy as np
from sklearn.neighbors import kneighbors_graph
from fancyimpute import MatrixFactorization, IterativeImputer

class Imputer:
    short_name: str

    def __init__(self, method=None, is_deterministic=True, in_sample=True):
        self.name = self.__class__.__name__
        self.method = method
        self.is_deterministic = is_deterministic
        self.in_sample = in_sample

    def fit(self, x, mask):
        if not self.in_sample:
            x_hat = np.where(mask, x, np.nan)
            return self.method.fit(x_hat)

    def predict(self, x, mask):
        x_hat = np.where(mask, x, np.nan)
        if self.in_sample:
            return self.method.fit_transform(x_hat)
        else:
            return self.method.transform(x_hat)

    def params(self):
        return dict()


class SpatialKNNImputer(Imputer):
    short_name = 'knn'

    def __init__(self, adj, k=20):
        super(SpatialKNNImputer, self).__init__()
        self.k = k
        # normalize sim between [0, 1]
        sim = (adj + adj.min()) / (adj.max() + adj.min())
        knns = kneighbors_graph(1 - sim,
                                n_neighbors=self.k,
                                include_self=False,
                                metric='precomputed').toarray()
        self.knns = knns

    def fit(self, x, mask):
        pass

    def predict(self, x, mask):
        x = np.where(mask, x, 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            y_hat = (x @ self.knns.T) / (mask @ self.knns.T)
        y_hat[~np.isfinite(y_hat)] = x.mean()
        return np.where(mask, x, y_hat)

    def params(self):
        return dict(k=self.k)


class MeanImputer(Imputer):
    short_name = 'mean'

    def fit(self, x, mask):
        d = np.where(mask, x, np.nan)
        self.means = np.nanmean(d, axis=0, keepdims=True)

    def predict(self, x, mask):
        if self.in_sample:
            d = np.where(mask, x, np.nan)
            means = np.nanmean(d, axis=0, keepdims=True)
        else:
            means = self.means
        return np.where(mask, x, means)


class MatrixFactorizationImputer(Imputer):
    short_name = 'mf'

    def __init__(self, rank=10, loss='mae', verbose=0):
        method = MatrixFactorization(rank=rank, loss=loss, verbose=verbose)
        super(MatrixFactorizationImputer, self).__init__(method, is_deterministic=False, in_sample=True)

    def params(self):
        return dict(rank=self.method.rank)


class MICEImputer(Imputer):
    short_name = 'mice'

    def __init__(self, max_iter=100, n_nearest_features=None, in_sample=True, verbose=False):
        method = IterativeImputer(max_iter=max_iter, n_nearest_features=n_nearest_features, verbose=verbose)
        is_deterministic = n_nearest_features is None
        super(MICEImputer, self).__init__(method, is_deterministic=is_deterministic, in_sample=in_sample)

    def params(self):
        return dict(max_iter=self.method.max_iter, k=self.method.n_nearest_features or -1)

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n', 'off'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y', 'on'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')