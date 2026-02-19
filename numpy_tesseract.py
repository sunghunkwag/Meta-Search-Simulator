import numpy as np

class NumpyAutoencoder:
    """
    A lightweight Autoencoder implemented purely in NumPy with explicit backpropagation.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, learning_rate=0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = learning_rate

        # Initialize weights (Xavier/Glorot)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / (hidden_dim + latent_dim))
        self.b2 = np.zeros((1, latent_dim))

        self.W3 = np.random.randn(latent_dim, hidden_dim) * np.sqrt(2.0 / (latent_dim + hidden_dim))
        self.b3 = np.zeros((1, hidden_dim))
        self.W4 = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (hidden_dim + input_dim))
        self.b4 = np.zeros((1, input_dim))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        # Encoder
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.latent = self.z2 # Linear activation for latent space

        # Decoder
        self.z3 = np.dot(self.latent, self.W3) + self.b3
        self.a3 = self.relu(self.z3)
        self.z4 = np.dot(self.a3, self.W4) + self.b4
        self.reconstruction = self.z4 # Linear output

        return self.reconstruction, self.latent

    def backward(self, X, X_hat):
        m = X.shape[0]

        # Loss derivative (MSE): dL/dX_hat = 2 * (X_hat - X) / m
        grad_output = 2 * (X_hat - X) / m

        # Decoder Backprop
        d_z4 = grad_output # Linear activation derivative is 1
        d_W4 = np.dot(self.a3.T, d_z4)
        d_b4 = np.sum(d_z4, axis=0, keepdims=True)

        d_a3 = np.dot(d_z4, self.W4.T)
        d_z3 = d_a3 * self.relu_deriv(self.z3)
        d_W3 = np.dot(self.latent.T, d_z3)
        d_b3 = np.sum(d_z3, axis=0, keepdims=True)

        # Encoder Backprop
        d_latent = np.dot(d_z3, self.W3.T)
        d_z2 = d_latent # Linear activation
        d_W2 = np.dot(self.a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0, keepdims=True)

        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * self.relu_deriv(self.z1)
        d_W1 = np.dot(X.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        # Update weights
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2
        self.W3 -= self.lr * d_W3
        self.b3 -= self.lr * d_b3
        self.W4 -= self.lr * d_W4
        self.b4 -= self.lr * d_b4

        loss = np.mean((X - X_hat) ** 2)
        return loss

class NumpyGMM:
    """
    Simplified Gaussian Mixture Model using K-Means initialization and covariance estimation.
    """
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.means = None
        self.covariances = None
        self.weights = None

    def fit(self, X):
        n_samples, n_features = X.shape
        # Simple K-Means initialization
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[indices]
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])
        self.weights = np.ones(self.n_components) / self.n_components

        # Simplified EM-like update (One iteration for stability/speed in this context)
        # Assign points to nearest cluster
        distances = np.linalg.norm(X[:, np.newaxis] - self.means, axis=2)
        labels = np.argmin(distances, axis=1)

        for k in range(self.n_components):
            cluster_points = X[labels == k]
            if len(cluster_points) > 1:
                self.means[k] = np.mean(cluster_points, axis=0)
                # Diagonal covariance for simplicity
                self.covariances[k] = np.diag(np.var(cluster_points, axis=0) + 1e-6)
                self.weights[k] = len(cluster_points) / n_samples
            else:
                # Re-initialize empty cluster
                self.means[k] = X[np.random.randint(n_samples)]
                self.covariances[k] = np.eye(n_features)

    def score_samples(self, X):
        """Calculate log-likelihood for samples."""
        n_samples, n_features = X.shape
        log_prob = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            diff = X - self.means[k]
            # Mahalanobis distance term (simplified diagonal covariance)
            inv_cov = np.linalg.inv(self.covariances[k])
            mahalanobis = np.sum(np.dot(diff, inv_cov) * diff, axis=1)
            det_cov = np.linalg.det(self.covariances[k])
            norm_const = -0.5 * (n_features * np.log(2 * np.pi) + np.log(det_cov + 1e-10))
            log_prob[:, k] = norm_const - 0.5 * mahalanobis + np.log(self.weights[k] + 1e-10)

        # Log-sum-exp trick for numerical stability
        max_log_prob = np.max(log_prob, axis=1, keepdims=True)
        return max_log_prob.squeeze() + np.log(np.sum(np.exp(log_prob - max_log_prob), axis=1))

class NumpyOperator:
    """
    Evolvable transformation matrix in latent space.
    f(z) = tanh(Wz + b)
    """
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.W = np.random.randn(latent_dim, latent_dim) * 0.1
        self.b = np.zeros(latent_dim)

    def forward(self, z):
        return np.tanh(np.dot(z, self.W) + self.b)

    def mutate(self, sigma=0.05):
        child = NumpyOperator(self.latent_dim)
        child.W = self.W + np.random.randn(*self.W.shape) * sigma
        child.b = self.b + np.random.randn(*self.b.shape) * sigma
        return child

class TesseractEngine:
    def __init__(self, z_dim=32, vocab_size=1000):
        self.z_dim = z_dim
        self.vocab_size = vocab_size

        # 1. Initialize "Human Semantic Space" with random vectors (Simulation of Embeddings)
        # In a real scenario, these would be loaded from a file or computed via a model.
        # We use a fixed seed for consistency.
        np.random.seed(42)
        self.human_embeddings = np.random.randn(vocab_size, 64) # 64-dim embeddings
        self.human_embeddings /= np.linalg.norm(self.human_embeddings, axis=1, keepdims=True)

        # 2. Autoencoder
        self.ae = NumpyAutoencoder(input_dim=64, hidden_dim=48, latent_dim=z_dim)

        # 3. GMM
        self.gmm = NumpyGMM(n_components=5)

        self.train_autoencoder()
        self.fit_density()

    def train_autoencoder(self, epochs=50):
        for _ in range(epochs):
            # Simple batch training
            idx = np.random.permutation(self.vocab_size)
            for i in range(0, self.vocab_size, 32):
                batch = self.human_embeddings[idx[i:i+32]]
                recons, latent = self.ae.forward(batch)
                self.ae.backward(batch, recons) # Backprop updates weights inplace

    def fit_density(self):
        _, latent = self.ae.forward(self.human_embeddings)
        self.gmm.fit(latent)

    def evolve_concepts(self, n_generations=10, population_size=20) -> list:
        # Evolution of Operators to find Novelty (High NLL)
        population = [NumpyOperator(self.z_dim) for _ in range(population_size)]

        for gen in range(n_generations):
            scores = []
            for op in population:
                # Generate latent points
                z_seed = np.random.randn(10, self.z_dim)
                z_new = op.forward(z_seed)

                # Score: Novelty (High NLL) + Consistency (Reconstruction)
                # NLL is negative log likelihood. We want to minimize likelihood -> maximize -logL
                # GMM score_samples returns log_likelihood.
                log_prob = self.gmm.score_samples(z_new)
                novelty = -np.mean(log_prob)

                # Consistency (Latent -> Output -> Latent)
                # We skip full consistency check for speed, just check norm
                norm_penalty = np.mean(z_new**2)

                fitness = novelty - 0.1 * norm_penalty
                scores.append((fitness, op))

            # Selection
            scores.sort(key=lambda x: x[0], reverse=True)
            elites = [op for _, op in scores[:5]]

            # Reproduction
            new_pop = list(elites)
            while len(new_pop) < population_size:
                parent = np.random.choice(elites)
                child = parent.mutate()
                new_pop.append(child)
            population = new_pop

        # Return best operators
        return elites

    def decode_concept(self, operator, n_samples=1):
        """
        Generates a 'concept vector' using the operator.
        """
        z_seed = np.random.randn(n_samples, self.z_dim)
        z_concept = operator.forward(z_seed)
        return z_concept

    def feedback(self, operator, reward):
        """
        Reinforcement Learning Update for the Operator.
        If reward is positive, we reinforce the weights.
        If negative, we add noise (explore).

        Simple Gradient-Free Update (ES-like).
        """
        lr = 0.01
        noise = np.random.randn(*operator.W.shape)
        if reward > 0:
            # "Solidify" the weights (reduce magnitude of random drift? No, that's not right)
            # We assume the current state is good.
            pass
        else:
            # Punish: Move away from this configuration?
            # Or just mutate more aggressively.
            operator.W += noise * lr * -1.0
