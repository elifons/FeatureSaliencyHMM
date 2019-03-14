# Hidden Markov Models
#
# Author: Ron Weiss <ronweiss@gmail.com>
#         Shiqiao Du <lucidfrontier.45@gmail.com>
# API changes: Jaques Grobler <jaquesgrobler@gmail.com>
# Modifications to create of the HMMLearn module: Gael Varoquaux
# More API changes: Sergei Lebedev <superbobry@gmail.com>
# FSHMM code: Elizabeth Fons <elifons@gmail.com>
#			  Alejandro Sztrajman <asztrajman@gmail.com>
# FSHMM algorithm from:
# Adams, Stephen & Beling, Peter & Cogill, Randy. (2016). 
# Feature Selection for Hidden Markov Models and Hidden Semi-Markov Models. 
"""
The :mod:`hmmlearn.hmm` module implements hidden Markov models.
"""

import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.mixture import (
    GMM, sample_gaussian,
    log_multivariate_normal_density,
    distribute_covar_matrix_to_match_covariance_type, _validate_covars)

from scipy.misc import logsumexp
from sklearn.base import BaseEstimator, _pprint
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

from base import _BaseHMM
from hmmlearn.utils import normalize, iter_from_X_lengths, normalize

from base import ConvergenceMonitor

__all__ = ["GMMHMM", "GaussianHMM", "MultinomialHMM", "GaussianFSHHM"]

COVARIANCE_TYPES = frozenset(("spherical", "diag", "full", "tied"))


class GaussianHMM(_BaseHMM):
    """Hidden Markov Model with Gaussian emissions.

    Parameters
    ----------
    n_components : int
        Number of states.

    covariance_type : string
        String describing the type of covariance parameters to
        use.  Must be one of

        * "spherical" --- each state uses a single variance value that
          applies to all features;
        * "diag" --- each state uses a diagonal covariance matrix;
        * "full" --- each state uses a full (i.e. unrestricted)
          covariance matrix;
        * "tied" --- all states use **the same** full covariance matrix.

        Defaults to "diag".

    min_covar : float
        Floor on the diagonal of the covariance matrix to prevent
        overfitting. Defaults to 1e-3.

    startprob_prior : array, shape (n_components, )
        Initial state occupation prior distribution.

    transmat_prior : array, shape (n_components, n_components)
        Matrix of prior transition probabilities between states.

    algorithm : string
        Decoder algorithm. Must be one of "viterbi" or "map".
        Defaults to "viterbi".

    random_state: RandomState or an int seed
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'm' for means and 'c' for covars. Defaults
        to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'm' for means and 'c' for covars.
        Defaults to all parameters.

    Attributes
    ----------
    n_features : int
        Dimensionality of the Gaussian emissions.

    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.

    means\_ : array, shape (n_components, n_features)
        Mean parameters for each state.

    covars\_ : array
        Covariance parameters for each state.

        The shape depends on ``covariance_type``::

            (n_components, )                        if 'spherical',
            (n_features, n_features)                if 'tied',
            (n_components, n_features)              if 'diag',
            (n_components, n_features, n_features)  if 'full'

    Examples
    --------
    >>> from hmmlearn.hmm import GaussianHMM
    >>> GaussianHMM(n_components=2)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    GaussianHMM(algorithm='viterbi',...
    """
    def __init__(self, n_components=1, covariance_type='diag',
                 min_covar=1e-3,
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 algorithm="viterbi", random_state=None,
                 n_iter=400, tol=1e-9, verbose=False,
        
                 params="stmc", init_params="stmc"):
        _BaseHMM.__init__(self, n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params)

        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight

    def _get_covars(self):
        """Return covars as a full matrix."""
        if self.covariance_type == 'full':
            return self._covars_
        elif self.covariance_type == 'diag':
            return np.array([np.diag(cov) for cov in self._covars_])
        elif self.covariance_type == 'tied':
            return np.array([self._covars_] * self.n_components)
        elif self.covariance_type == 'spherical':
            return np.array(
                [np.eye(self.n_features) * cov for cov in self._covars_])

    def _set_covars(self, covars):
        self._covars_ = np.asarray(covars).copy()
        covars_ = property(_get_covars, _set_covars)

    def _check(self):
        super(GaussianHMM, self)._check()

        self.means_ = np.asarray(self.means_)
        self.n_features = self.means_.shape[1]
        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError('covariance_type must be one of {0}'
                             .format(COVARIANCE_TYPES))
        _validate_covars(self._covars_, self.covariance_type,
                         self.n_components)

    def _init(self, X, lengths=None):
        super(GaussianHMM, self)._init(X, lengths=lengths)

        _, n_features = X.shape
        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (n_features, self.n_features))

        self.n_features = n_features
        if 'm' in self.init_params or not hasattr(self, "means_"): #paper: mu initialized randomly
            kmeans = cluster.KMeans(n_clusters=self.n_components,
                                    random_state=self.random_state)
            kmeans.fit(X)
            self.means_ = kmeans.cluster_centers_
        if 'c' in self.init_params or not hasattr(self, "covars_"): #paper: sigma initialized with 4
            cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
            if not cv.shape:
                cv.shape = (1, 1)
            self._covars_ = distribute_covar_matrix_to_match_covariance_type(
                cv, self.covariance_type, self.n_components).copy()
            self._covars_ = np.ones(self._covars_.shape)*4.0 #villavilla

    def _compute_log_likelihood(self, X):
        return log_multivariate_normal_density(
            X, self.means_, self._covars_, self.covariance_type)

    def _generate_sample_from_state(self, state, random_state=None):
        if self.covariance_type == 'tied':
            cv = self._covars_
        else:
            cv = self._covars_[state]
        return sample_gaussian(self.means_[state], cv, self.covariance_type,
                               random_state=random_state)

    def _initialize_sufficient_statistics(self):
        stats = super(GaussianHMM, self)._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        if self.covariance_type in ('tied', 'full'):
            stats['obs*obs.T'] = np.zeros((self.n_components, self.n_features,
                                           self.n_features))
        # print(stats)
        return stats

    def _accumulate_sufficient_statistics(self, stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice):
        super(GaussianHMM, self)._accumulate_sufficient_statistics(
            stats, obs, framelogprob, posteriors, fwdlattice, bwdlattice)

        if 'm' in self.params or 'c' in self.params:
            stats['post'] += posteriors.sum(axis=0)
            stats['obs'] += np.dot(posteriors.T, obs)

        if 'c' in self.params:
            if self.covariance_type in ('spherical', 'diag'):
                stats['obs**2'] += np.dot(posteriors.T, obs ** 2)
            elif self.covariance_type in ('tied', 'full'):
                # posteriors: (nt, nc); obs: (nt, nf); obs: (nt, nf)
                # -> (nc, nf, nf)
                stats['obs*obs.T'] += np.einsum(
                    'ij,ik,il->jkl', posteriors, obs, obs)

    def _do_mstep(self, stats):
        super(GaussianHMM, self)._do_mstep(stats)

        means_prior = self.means_prior
        means_weight = self.means_weight

        # TODO: find a proper reference for estimates for different
        #       covariance models.
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        denom = stats['post'][:, np.newaxis]
        if 'm' in self.params:
            self.means_ = ((means_weight * means_prior + stats['obs'])
                           / (means_weight + denom))

        if 'c' in self.params:
            covars_prior = self.covars_prior
            covars_weight = self.covars_weight
            meandiff = self.means_ - means_prior

            if self.covariance_type in ('spherical', 'diag'):
                cv_num = (means_weight * meandiff**2
                          + stats['obs**2']
                          - 2 * self.means_ * stats['obs']
                          + self.means_**2 * denom)
                cv_den = max(covars_weight - 1, 0) + denom
                self._covars_ = \
                    (covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
                if self.covariance_type == 'spherical':
                    self._covars_ = np.tile(
                        self._covars_.mean(1)[:, np.newaxis],
                        (1, self._covars_.shape[1]))
            elif self.covariance_type in ('tied', 'full'):
                cv_num = np.empty((self.n_components, self.n_features,
                                  self.n_features))
                for c in range(self.n_components):
                    obsmean = np.outer(stats['obs'][c], self.means_[c])

                    cv_num[c] = (means_weight * np.outer(meandiff[c],
                                                         meandiff[c])
                                 + stats['obs*obs.T'][c]
                                 - obsmean - obsmean.T
                                 + np.outer(self.means_[c], self.means_[c])
                                 * stats['post'][c])
                cvweight = max(covars_weight - self.n_features, 0)
                if self.covariance_type == 'tied':
                    self._covars_ = ((covars_prior + cv_num.sum(axis=0)) /
                                     (cvweight + stats['post'].sum()))
                elif self.covariance_type == 'full':
                    self._covars_ = ((covars_prior + cv_num) /
                                     (cvweight + stats['post'][:, None, None]))

#init values
#self.startprob_ -> pi_i (i:estado)
#self.transmat_ -> a_ij (i,j:estado)
#self.means_ -> mu_il (i:estado, l:feature/serie)
#self._covars_ -> sigma2_il (i:estado, l:feature/serie)
#self.rho_ -> rho_l (l:feature/serie)
#self.epsilon_ -> epsilon_l (l:feature/serie)
#self.tau_ -> tau_l (l:feature/serie)

#self.n_features = X.shape[1], con dimension L
#self.n_components -> number of states

#FIXME: revisar que cuando le asigné tau o std() a una variable esté bien eso, y que no tendría que haberle asignado _covars_ o algo distinto.
#FIXME: revisar que las cuentas finales con np.dot(np.sum()) y np.sum(np.sum()) hagan lo que tienen que hacer.


def gaussiana(x, mu, sigma2):
    return (1.0/np.sqrt(2*np.pi*sigma2))*np.exp(((x - mu)**2)/(2*sigma2))

class GaussianFSHMM(GaussianHMM):
    def __init__(self, k, **kwargs):
        super(GaussianFSHMM, self).__init__(**kwargs)
        self.k_factor_ = k

    def init_values_FS(self, X, epsilon=None, tau=None, rho=None): #default initialization is for the first example of the paper
        self.rho_ = np.ones(self.n_features)*0.5 if (rho is None) else rho

        self.epsilon_ = pd.DataFrame(X).mean().values if (epsilon is None) else epsilon

        self.tau_ = pd.DataFrame(X).std().values if (tau is None) else tau

    def select_hyperparams(self, X): #pass hyperparameters as arguments of this function
        self.p_ = np.ones(self.n_components)*2
        self.a_ = np.ones((self.n_components, self.n_components))*2

        self.b_ = pd.DataFrame(X).mean().values # self.b_ = X.mean().values

        self.m_ = self.means_.copy()
        for l in range(self.n_features):
            self.m_[0, l] = self.b_[l] - self.tau_[l]
            self.m_[1, l] = self.b_[l] + self.tau_[l]

        self.s_ = np.ones(self.m_.shape)*0.5
        self.zeta_ = np.ones(self.m_.shape)*0.5
        self.eta_ = np.ones(self.m_.shape)*0.5

        self.c_ = np.ones(self.n_features)*1.0
        self.nu_ = np.ones(self.n_features)*0.5
        self.psi_ = np.ones(self.n_features)*0.5
        self.k_ = np.ones(self.n_features)*self.k_factor_

    def compute_FS_ESTEP(self, X, gamma):
        I = self.n_components
        L = self.n_features
        T = X.shape[0]
        self.uilt = np.zeros((I, L, T))
        self.vilt = np.zeros((I, L, T))
        for i in range(I):
            for l in range(L):
                for t in range(T):
                    eilt = self.rho_[l]*gaussiana(X[t, l], self.means_[i, l], self._covars_[i, l])
                    hilt = (1.0-self.rho_[l])*gaussiana(X[t, l], self.epsilon_[l], self.tau_[l])
                    gilt = eilt + hilt
                    uilt = gamma[t, i]*eilt/gilt #tiene este orden gamma?
                    vilt = gamma[t, i] - uilt
                    self.uilt[i, l, t] = uilt
                    self.vilt[i, l, t] = vilt
        



    def fit(self, X, lengths=None):
        """Estimate model parameters.

        An initialization step is performed before entering the
        EM algorithm. If you want to avoid this step for a subset of
        the parameters, pass proper ``init_params`` keyword argument
        to estimator's constructor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)
        self._init(X, lengths=lengths)
        self._check()
        self.init_values_FS(X)
        self.select_hyperparams(X)

        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)
        for iter in range(self.n_iter):
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0
            for i, j in iter_from_X_lengths(X, lengths):
                framelogprob = self._compute_log_likelihood(X[i:j])
                logprob, fwdlattice = self._do_forward_pass(framelogprob)
                curr_logprob += logprob
                bwdlattice = self._do_backward_pass(framelogprob)
                posteriors = self._compute_posteriors(fwdlattice, bwdlattice) #posteriors <- gamma
                self._accumulate_sufficient_statistics(
                    stats, X[i:j], framelogprob, posteriors, fwdlattice,
                    bwdlattice)
                self.compute_FS_ESTEP(X, posteriors)

            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.
            self._do_mstep(X, stats)

            self.monitor_.report(curr_logprob)
            if self.monitor_.converged:
                break

        return self

    def _do_mstep(self, X, stats):
        """Performs the M-step of EM algorithm.

        Parameters
        ----------
        stats : dict
            Sufficient statistics updated from all available samples.
        """
        # The ``np.where`` calls guard against updating forbidden states
        # or transitions in e.g. a left-right HMM.
        I = self.n_components
        L = self.n_features
        T = X.shape[0]

        if 's' in self.params:
            startprob_ = self.startprob_prior - 1.0 + stats['start'] #dimensions? #estos asumimos que estan bien
            self.startprob_ = np.where(self.startprob_ == 0.0,
                                       self.startprob_, startprob_)
            normalize(self.startprob_)
        if 't' in self.params:
            transmat_ = self.transmat_prior - 1.0 + stats['trans'] #dimensions? #estos asumimos que estan bien
            self.transmat_ = np.where(self.transmat_ == 0.0,
                                      self.transmat_, transmat_)
            normalize(self.transmat_, axis=1)

        means_prior = self.means_prior
        means_weight = self.means_weight

        # TODO: find a proper reference for estimates for different
        #       covariance models.
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445
        denom = stats['post'][:, np.newaxis]
        if 'm' in self.params:
            for i in range(I):
                for l in range(L):
                    sil2 = self.s_[i,l]**2
                    sigmail2 = self._covars_[i,l]
                    term1 = sil2*np.dot(self.uilt[i, l, :], X[:,l])
                    num = term1 + self.m_[i, l]*sigmail2
                    den = sil2*np.sum(self.uilt[i, l, :]) + sigmail2
                    self.means_[i, l] = num/den
            #self.means_ = ((means_weight * means_prior + stats['obs'])
            #              / (means_weight + denom))

        if 'c' in self.params:
            covars_prior = self.covars_prior
            covars_weight = self.covars_weight
            meandiff = self.means_ - means_prior

            if self.covariance_type in ('spherical', 'diag'):
                for i in range(I):
                    for l in range(L):
                        term1 = np.dot(self.uilt[i, l, :], (X[:,l] - self.means_[i, l])**2)
                        num = term1 + 2 * self.eta_[i, l]
                        den = np.sum(self.uilt[i, l, :]) + 2*(self.zeta_[i, l] + 1.0)
                        self._covars_[i, l] = num/den

        for l in range(L):
            cl2 = self.c_[l]**2
            bl = self.b_[l]
            taul2 = self.tau_[l]**2
            psil = self.psi_[l]
            nul = self.nu_[l]
            epsilonl = self.epsilon_[l]
            kl = self.k_[l]
            hatT = T + 1 + kl

            epsilonl_num = cl2*np.dot(np.sum(self.vilt[:, l, :], axis=0), X[:, l]) + taul2*bl #????
            epsilonl_den = cl2*np.sum(np.sum(self.vilt[:, l, :], axis=0), axis=-1) + taul2
            self.epsilon_[l] = epsilonl_num / epsilonl_den

            taul_num = np.dot(np.sum(self.vilt[:, l, :], axis=0), (X[:, l] - epsilonl)**2) + 2*psil #????
            taul_den = np.sum(np.sum(self.vilt[:, l, :], axis=0), axis=-1) + 2 * (nul + 1.0)
            self.tau_[l] = np.sqrt(taul_num / taul_den)

            self.rho_[l] = (hatT - np.sqrt(hatT**2 - 4*kl*np.sum(np.sum(self.uilt[:, l, :], axis=0), axis=-1))) / (2*kl)




