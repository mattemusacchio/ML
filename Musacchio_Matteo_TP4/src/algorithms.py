import numpy as np

def kmeans(X, k, max_iters=100, tol=1e-4):
    # Inicialización aleatoria de centroides
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx]

    for _ in range(max_iters):
        # Asignar puntos al centroide más cercano
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Calcular nuevos centroides
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        
        # Verificar convergencia
        if np.linalg.norm(centroids - new_centroids) < tol:
            break
        centroids = new_centroids

    # Calcular L (suma de distancias cuadradas de cada punto a su centroide)
    final_distances = np.linalg.norm(X - centroids[labels], axis=1)
    L = np.sum(final_distances)

    return labels, centroids, L

def multivariate_gaussian(X, means, covs):
    N, D = X.shape
    K = means.shape[0]
    probs = np.zeros((N, K))

    for k in range(K):
        mean = means[k]
        cov = covs[k]
        cov_det = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)
        norm_const = 1.0 / (np.power(2 * np.pi, D / 2) * np.sqrt(cov_det))

        diffs = X - mean
        exponents = np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs)
        probs[:, k] = norm_const * np.exp(-0.5 * exponents)

    return probs

def gmm(X, k, max_iters=50, tol=1e-3):
    N, D = X.shape

    # Inicialización con K-means
    labels, means, _ = kmeans(X, k)
    pis = np.array([np.mean(labels == i) for i in range(k)])
    covs = []

    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 1:
            diff = cluster_points - cluster_points.mean(axis=0)
            cov = (diff.T @ diff) / len(cluster_points)
        else:
            cov = np.eye(D)
        covs.append(cov + 1e-6 * np.eye(D))  # Regularización

    covs = np.array(covs)
    prev_ll = None

    for _ in range(max_iters):
        # E-step
        probs = multivariate_gaussian(X, means, covs)
        weighted_probs = probs * pis
        gamma = weighted_probs / np.sum(weighted_probs, axis=1, keepdims=True)

        # M-step
        Nks = np.sum(gamma, axis=0)
        pis = Nks / N
        means = (gamma.T @ X) / Nks[:, None]
        covs = np.zeros((k, D, D))

        for i in range(k):
            diff = X - means[i]
            covs[i] = (gamma[:, i][:, None] * diff).T @ diff / Nks[i]
            covs[i] += 1e-6 * np.eye(D)  # Regularización

        # Log-likelihood
        probs = multivariate_gaussian(X, means, covs)
        weighted_probs = probs * pis
        log_likelihood = np.sum(np.log(np.sum(weighted_probs, axis=1) + 1e-10))

        if prev_ll is not None and abs(log_likelihood - prev_ll) < tol:
            break
        prev_ll = log_likelihood

    final_labels = np.argmax(gamma, axis=1)
    return final_labels, means, log_likelihood

import numpy as np

def dbscan(X, eps, min_pts):
    N = len(X)
    labels = np.full(N, -1)  # Inicialmente todo es ruido
    visited = np.zeros(N, dtype=bool)
    cluster_id = 0

    def region_query(i):
        # Devuelve índices de los puntos dentro de eps de X[i]
        dists = np.linalg.norm(X - X[i], axis=1)
        return np.where(dists <= eps)[0]

    def expand_cluster(i, neighbors):
        nonlocal cluster_id
        labels[i] = cluster_id
        j = 0
        while j < len(neighbors):
            n_i = neighbors[j]
            if not visited[n_i]:
                visited[n_i] = True
                new_neighbors = region_query(n_i)
                if len(new_neighbors) >= min_pts:
                    neighbors = np.append(neighbors, new_neighbors)
            if labels[n_i] == -1:
                labels[n_i] = cluster_id
            j += 1

    for i in range(N):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = region_query(i)
        if len(neighbors) >= min_pts:
            expand_cluster(i, neighbors)
            cluster_id += 1

    return labels

def pca(X, n_components):
    """
    Realiza PCA sobre los datos X y devuelve la proyección y la reconstrucción.
    
    Parámetros:
        X: np.ndarray, matriz de datos de forma (n_samples, n_features)
        n_components: int, número de componentes principales a conservar

    Retorna:
        X_reduced: np.ndarray, datos proyectados (n_samples, n_components)
        X_reconstructed: np.ndarray, datos reconstruidos (n_samples, n_features)
    """
    # Centrar datos
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Calcular matriz de covarianza
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Autovalores y autovectores
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # Ordenar de mayor a menor
    idx_sorted = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx_sorted[:n_components]]

    # Proyección
    X_reduced = X_centered @ eigvecs

    # Reconstrucción
    X_reconstructed = X_reduced @ eigvecs.T + X_mean

    return X_reduced, X_reconstructed
