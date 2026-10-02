import numpy as np
from scipy.linalg import cholesky, cho_factor, cho_solve, eigh
from scipy.stats import qmc

from .erroneousPreference import erroneousPreference
from inference import slice_sampler
from kernel import RBF
from utility import paramz


class PreferenceOptimizer:

    def __init__(self, bounds, q=2, lengthscale=0.3, variance=4.0, n_candidates=1024,
                 nsamples=5000, tune=1000, tune_warm=200, n_mc=256,
                 optimize_hyperparams=False, seed=None):
        """
        Preferential Bayesian optimisation: takes pairwise preferences (duels) between points and
        suggests the next points to compare.

        The utility is modelled with the erroneousPreference GP (probit likelihood, so inconsistent
        answers are allowed), fitted only on the points that have been compared. The next query
        maximises qEUBO, E[max of the utilities of the q options], over a Sobol candidate set.

        Preferences can be added any number of times. Each fit uses all the preferences so far,
        which is the same posterior as updating the previous GP with the new ones; the sampler is
        warm-started from the end of the previous chain.

        The samplers of this library use the global NumPy random state: call np.random.seed
        before using the optimizer for fully reproducible results.

        :param bounds: lower and upper bound of each input dimension
        :type bounds: array (D x 2)
        :param q: number of options suggested for each query (2 is a duel, EUBO)
        :type q: integer
        :param lengthscale: RBF lengthscale in the unit cube (scalar or one value per dimension)
        :param variance: RBF variance; the probit noise is 1, so larger values trust each answer more
        :param n_candidates: size of the Sobol candidate set (rounded up to a power of 2)
        :param nsamples: number of posterior samples
        :param tune: burn-in samples of the first fit
        :param tune_warm: burn-in samples of the warm-started fits
        :param n_mc: number of posterior samples used to evaluate qEUBO
        :param optimize_hyperparams: optimise the kernel hyperparameters at every fit
        :param seed: seed for the Sobol candidates and the acquisition samples
        """
        self.bounds = np.atleast_2d(np.asarray(bounds, float))
        if self.bounds.shape[1] != 2 or np.any(self.bounds[:, 1] <= self.bounds[:, 0]):
            raise ValueError("bounds must be a (D x 2) array of [lower, upper] with upper > lower")
        self.D = self.bounds.shape[0]
        self.q = q
        self.n_candidates = n_candidates
        self.nsamples = nsamples
        self.tune = tune
        self.tune_warm = tune_warm
        self.n_mc = n_mc
        self.optimize_hyperparams = optimize_hyperparams
        self.params = {'lengthscale': {'value': lengthscale * np.ones(self.D, float),
                                       'range': np.vstack([[0.05, 5.0]] * self.D),
                                       'transform': paramz.logexp()},
                       'variance': {'value': np.array([variance], float),
                                    'range': np.vstack([[0.1, 50.0]]),
                                    'transform': paramz.logexp()}}
        self.pairs = []  # [winner, loser] indices over the rows of X
        self._Xu = np.zeros((0, self.D))  # compared points in the unit cube
        self._model = None
        self._latent = np.zeros(0)  # last state of the sampler (one value per pair)
        self._fitted_pairs = 0
        self._rng = np.random.default_rng(seed)

    # ---- scaling ----
    def _to_unit(self, x):
        return (np.asarray(x, float) - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])

    def _from_unit(self, z):
        return self.bounds[:, 0] + np.asarray(z, float) * (self.bounds[:, 1] - self.bounds[:, 0])

    @property
    def X(self):
        """Compared points (n x D) in the original units; pairs index its rows."""
        return self._from_unit(self._Xu)

    @property
    def model(self):
        """The fitted erroneousPreference model (inputs in the unit cube), None without data."""
        self.fit()
        return self._model

    # ---- data ----
    def _index_of(self, z):
        if self._Xu.shape[0] > 0:
            hit = np.flatnonzero(np.all(np.isclose(self._Xu, z, atol=1e-9), axis=1))
            if hit.size > 0:
                return int(hit[0])
        self._Xu = np.vstack([self._Xu, z])
        return self._Xu.shape[0] - 1

    def add_preferences(self, winners, losers):
        """
        Add one or more duels. Can be called any number of times.

        :param winners: preferred point of each duel, in the original units
        :type winners: array (D,) or (n x D)
        :param losers: other point of each duel
        :type losers: array (D,) or (n x D)
        """
        winners = np.atleast_2d(np.asarray(winners, float))
        losers = np.atleast_2d(np.asarray(losers, float))
        if winners.shape != losers.shape or winners.shape[1] != self.D:
            raise ValueError("winners and losers must both have shape (D,) or (n x D) with D=%d" % self.D)
        Zw, Zl = self._to_unit(winners), self._to_unit(losers)
        if np.any(np.hstack([Zw, Zl]) < -1e-9) or np.any(np.hstack([Zw, Zl]) > 1 + 1e-9):
            raise ValueError("points must lie inside the bounds")
        if np.any(np.all(np.isclose(Zw, Zl, atol=1e-9), axis=1)):
            raise ValueError("the two points of a duel must be different")
        for zw, zl in zip(Zw, Zl):
            self.pairs.append([self._index_of(zw), self._index_of(zl)])

    # ---- model ----
    def _sample(self, model, x0, tune):
        # same as erroneousPreference.sample, but the sampler starts from x0 and its last state is returned
        xi, Ω, Δ, γ, Γ = model._compute_SkewGP_posterior_params()
        iω = np.diag(1 / np.sqrt(np.diag(Ω)))
        Ω_c = np.linalg.multi_dot([iω, Ω, iω])  # correlation matrix
        L = cholesky(Γ + model.jitter * np.eye(Γ.shape[0]), lower=True)
        M = cho_solve((L, True), Δ.T).T
        M1 = Ω_c - np.linalg.multi_dot([M, Δ.T])
        M1 = 0.5 * (M1 + M1.T) + model.jitter * np.identity(M1.shape[0])
        L = cholesky(M1, lower=True)
        points1 = np.dot(L, np.random.randn(M1.shape[0], self.nsamples))
        # liness
        A = np.eye(Γ.shape[0])
        b = -γ
        latent = slice_sampler.liness_step(x0, A, b, np.zeros(Γ.shape[0]),
                                           Γ, nsamples=self.nsamples, tune=tune,
                                           progress=False)
        model.samples = xi + np.dot(np.diag(np.sqrt(np.diag(Ω))), points1 + M @ latent.T)
        return latent[-1]

    def fit(self):
        """Fit the model on all the preferences so far (does nothing if they have not changed)."""
        if len(self.pairs) == self._fitted_pairs:
            return
        data = {"Pairs": self.pairs, "X": self._Xu}
        model = erroneousPreference(data, RBF, self.params)
        if self.optimize_hyperparams:
            model.optimize_hyperparams(num_restarts=1)
            self.params = model.params  # next optimisation starts from these values
        # warm start: previous state of the sampler, any positive value is feasible for the new pairs
        warm = self._latent.size > 0
        x0 = np.hstack([self._latent, np.ones(len(self.pairs) - self._latent.size)])[:, None]
        self._latent = self._sample(model, x0, self.tune_warm if warm else self.tune)
        self._model = model
        self._fitted_pairs = len(self.pairs)

    def _conditional(self, Z):
        # f(Z) | f(X) ~ N(A.T f(X), C)
        model = self._model
        Kxx = RBF(self._Xu, self._Xu, model.params) + model.jitter * np.eye(self._Xu.shape[0])
        Kxz = RBF(self._Xu, Z, model.params)
        A = cho_solve(cho_factor(Kxx, lower=True), Kxz)
        return A, Kxz

    def predict(self, points):
        """
        Posterior mean and standard deviation of the utility, including the GP conditional variance.

        :param points: test points in the original units
        :type points: array (n x D)
        :returns mean (n,), std (n,)
        """
        self.fit()
        Z = self._to_unit(np.atleast_2d(points))
        var = self.params['variance']['value'][0]
        if self._model is None:
            return np.zeros(Z.shape[0]), np.sqrt(var) * np.ones(Z.shape[0])
        A, Kxz = self._conditional(Z)
        mean_samples = A.T @ self._model.samples
        cond_var = np.maximum(var - np.sum(Kxz * A, axis=0), 0.0)
        return mean_samples.mean(axis=1), np.sqrt(mean_samples.var(axis=1) + cond_var)

    def best(self):
        """
        :returns the compared point with the highest posterior mean (original units) and its mean
        """
        self.fit()
        if self._model is None:
            raise ValueError("no preferences have been added yet")
        mu = self._model.samples.mean(axis=1)
        i = int(np.argmax(mu))
        return self.X[i], mu[i]

    # ---- acquisition ----
    def _sobol(self, n):
        sampler = qmc.Sobol(d=self.D, scramble=True, seed=self._rng)
        return sampler.random_base2(int(np.ceil(np.log2(max(n, 2)))))

    def _joint_samples(self, Z):
        # joint posterior samples of the utility at Z (len(Z) x n_mc)
        A, Kxz = self._conditional(Z)
        cols = self._rng.choice(self._model.samples.shape[1], size=self.n_mc, replace=False)
        C = RBF(Z, Z, self._model.params) - Kxz.T @ A
        w, V = eigh(0.5 * (C + C.T))
        L = V * np.sqrt(np.maximum(w, 0.0))
        return A.T @ self._model.samples[:, cols] + L @ self._rng.standard_normal((Z.shape[0], self.n_mc))

    def suggest(self, q=None):
        """
        Next points to compare: the q candidates that maximise qEUBO = E[max_i f(x_i)].
        The candidates are a fresh Sobol set plus the points already compared.
        Without data the first Sobol points are returned.

        :param q: number of options (default: the value given at construction)
        :returns (q x D) array in the original units
        """
        q = self.q if q is None else q
        self.fit()
        if self._model is None:
            return self._from_unit(self._sobol(q)[:q])
        Z = np.vstack([self._sobol(self.n_candidates), self._Xu])
        F = self._joint_samples(Z)
        if q == 1:
            return self._from_unit(Z[[int(np.argmax(F.mean(axis=1)))]])
        # best pair, exact over all the candidate pairs
        best_val, chosen = -np.inf, None
        for start in range(0, Z.shape[0], 32):
            block = np.maximum(F[start:start + 32, None, :], F[None, :, :]).mean(axis=2)
            block[np.arange(block.shape[0]), start + np.arange(block.shape[0])] = -np.inf
            i, j = np.unravel_index(np.argmax(block), block.shape)
            if block[i, j] > best_val:
                best_val, chosen = block[i, j], [start + i, j]
        # further options are added greedily
        current = F[chosen].max(axis=0)
        while len(chosen) < q:
            gain = np.maximum(current, F).mean(axis=1)
            gain[chosen] = -np.inf
            k = int(np.argmax(gain))
            chosen.append(k)
            current = np.maximum(current, F[k])
        return self._from_unit(Z[chosen])
