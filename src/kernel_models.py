import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import RidgeClassifier
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted


def polynomial(x, y, degree=2, gamma=0.1, coef0=0.1):
    return (gamma * np.dot(x, y) + coef0) ** degree


def rbf(x, y, degree=2, gamma=0.1, coef0=0.1):
    return float(np.exp(-gamma * np.sum((x - y) ** 2)))


def hyperbolic(x, y, degree=2, gamma=0.1, coef0=0.1):
    return float(np.tanh(gamma * np.dot(x, y) + coef0))


def triangle(x, y, degree=2, gamma=0.1, coef0=0.1):
    d = np.linalg.norm(x - y)
    return float(max(0.0, 1.0 - d / max(gamma, 1e-12)))


def radial_basic(x, y, degree=2, gamma=0.1, coef0=0.1):
    return float(np.sum(np.exp(-gamma * (x - y) ** 2)) ** degree)


def rquadratic(x, y, degree=2, gamma=0.1, coef0=0.1):
    d2 = np.sum((x - y) ** 2)
    a = max(abs(coef0), 1e-8)
    return float(1.0 - d2 / (d2 + a))


def canberra(x, y, degree=2, gamma=0.1, coef0=0.1):
    den = np.abs(x) + np.abs(y)
    term = np.divide(np.abs(x-y), den, out=np.zeros_like(x, dtype=float), where=den != 0)
    return float(1.0 - gamma * np.mean(term))


def truncated(x, y, degree=2, gamma=0.1, coef0=0.1):
    d = np.abs(x-y)
    return float(np.mean(np.maximum(1.0 - d / max(gamma, 1e-12), 0.0)))


def linear(x, y, degree=2, gamma=0.1, coef0=0.1):
    return float(np.dot(x, y))

KERNEL_FUNCTIONS = {
    'linear': linear,
    'poly': polynomial,
    'rbf': rbf,
    'hyperbolic': hyperbolic,
    'triangle': triangle,
    'radial_basic': radial_basic,
    'rquadratic': rquadratic,
    'canberra': canberra,
    'truncated': truncated,
}


def gram_matrix(X, Y, kernel_name, degree=2, gamma=0.1, coef0=0.1):
    fn = KERNEL_FUNCTIONS[kernel_name]
    out = np.empty((len(X), len(Y)), dtype=float)
    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            out[i, j] = fn(x, y, degree, gamma, coef0)
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)


def kernel_callable(kernel_name, degree=2, gamma=0.1, coef0=0.1):
    # Callable matricial para SVC.
    def call(X, Y):
        return gram_matrix(X, Y, kernel_name, degree, gamma, coef0)
    return call

def scalar_kernel_callable(kernel_name, degree=2, gamma=0.1, coef0=0.1):
    # Callable escalar para Nystroem/pairwise_kernels.
    fn=KERNEL_FUNCTIONS[kernel_name]
    def call(x, y):
        return fn(np.asarray(x), np.asarray(y), degree, gamma, coef0)
    return call


class KSVC(SVC):
    """SVC compatible con los kernels alternativos del experimento."""
    def __init__(self, C=1.0, kernel='rbf', degree=2, gamma=0.1, coef0=0.1,
                 shrinking=True, probability=False, tol=1e-3, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1,
                 decision_function_shape='ovr', random_state=2021):
        super().__init__(C=C, kernel=kernel, degree=degree, gamma=gamma,
                         coef0=coef0, shrinking=shrinking, probability=probability,
                         tol=tol, cache_size=cache_size, class_weight=class_weight,
                         verbose=verbose, max_iter=max_iter,
                         decision_function_shape=decision_function_shape,
                         random_state=random_state)
        self._requested_kernel = kernel

    def fit(self, X, y, sample_weight=None):
        if self.kernel in KERNEL_FUNCTIONS and self.kernel not in {'linear', 'poly', 'rbf'}:
            self.kernel = kernel_callable(self.kernel, self.degree, self.gamma, self.coef0)
        return super().fit(X, y, sample_weight=sample_weight)

    def decision_function(self, X):
        score = super().decision_function(X)
        if np.ndim(score) == 1:
            return np.column_stack([-score, score])
        return score


class KANNC(MLPClassifier):
    """MLP precedido por un mapa de características de Nystroem."""
    def __init__(self, hidden_layer_sizes=(100,), activation='identity', solver='adam',
                 alpha=0.0001, batch_size='auto', learning_rate='constant',
                 learning_rate_init=0.001, max_iter=1000, shuffle=True,
                 random_state=2021, tol=1e-4, early_stopping=True,
                 validation_fraction=0.1, n_iter_no_change=10, kernel='rbf',
                 degree=2, gamma=0.1, coef0=0.1, n_components=100):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                         solver=solver, alpha=alpha, batch_size=batch_size,
                         learning_rate=learning_rate, learning_rate_init=learning_rate_init,
                         max_iter=max_iter, shuffle=shuffle, random_state=random_state,
                         tol=tol, early_stopping=early_stopping,
                         validation_fraction=validation_fraction,
                         n_iter_no_change=n_iter_no_change)
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.n_components = n_components
        self.feature_map_ = None

    def fit(self, X, y):
        self.feature_map_ = None
        if self.kernel != 'linear':
            n_components = min(self.n_components, max(2, len(X)))
            self.feature_map_ = Nystroem(
                kernel=scalar_kernel_callable(self.kernel, self.degree, self.gamma, self.coef0),
                n_components=n_components,
                random_state=self.random_state
            )
            X = self.feature_map_.fit_transform(X)
        return super().fit(X, y)

    def _map(self, X):
        return self.feature_map_.transform(X) if self.feature_map_ is not None else X

    def predict(self, X):
        return super().predict(self._map(X))

    def predict_proba(self, X):
        return super().predict_proba(self._map(X))

    def decision_function(self, X):
        p = self.predict_proba(X)
        return p


class KRidgeClassifier(RidgeClassifier):
    """RidgeClassifier en formulación dual usando una matriz kernel."""
    def __init__(self, alpha=1.0, kernel='rbf', degree=2, gamma=0.1,
                 coef0=0.1, random_state=2021):
        super().__init__(alpha=alpha)
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_, encoded = np.unique(y, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError('Se necesitan al menos dos clases.')
        self.X_fit_ = X
        K = gram_matrix(X, X, self.kernel, self.degree, self.gamma, self.coef0)
        target = -np.ones((len(y), len(self.classes_)))
        target[np.arange(len(y)), encoded] = 1.0
        A = K + self.alpha * np.eye(len(K))
        try:
            self.dual_coef_ = np.linalg.solve(A, target)
        except np.linalg.LinAlgError:
            self.dual_coef_ = np.linalg.pinv(A) @ target
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X):
        check_is_fitted(self, ['dual_coef_', 'X_fit_'])
        X = check_array(X)
        K = gram_matrix(X, self.X_fit_, self.kernel, self.degree, self.gamma, self.coef0)
        return K @ self.dual_coef_

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        scores = self.decision_function(X)
        scores -= scores.max(axis=1, keepdims=True)
        e = np.exp(scores)
        return e / e.sum(axis=1, keepdims=True)
