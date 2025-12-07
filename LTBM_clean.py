import numpy as np
from scipy.special import digamma, gammaln

import time


class LTBM:
    def __init__(self, A, W, dictionnary, alpha, Q, L, K, tol=1e-3, YX=None, seed=2023):
        """
        @param A: matrice d'incidence
        @param W: dict ; clé: (i, j), ou A[i, j] != 0 ; valeur: liste de liste de mots (! encodés en id 0, 1,... !)
        @param dictionnary: list ; liste de tous les mots
        @param alpha: vecteur des paramètres de concentration de Dirichlet de thehta (fixé dans le modele)

        @param Q: nombre de cluster ligne
        @param L: nombre de cluster colonne
        @param K: nombre de sujets

        @param tol: tolérance pour la convergence
        @param YX: tuple (Y, X) pour l'initialisation des assignations de cluster
        @param seed: graine pour la reproductibilité
        """
        np.random.seed(seed)

        self.tol = tol

        self.A = A
        self.T = [(i, j) for i, j in zip(*np.where(A != 0))]  # entrées non nulles
        self.W = W

        self.V = len(dictionnary)  # taille du vocabulaire

        self.Q = Q
        self.L = L
        self.K = K

        self.alpha = alpha
        self.gln_alpha = gammaln(self.alpha)
        self.gln_alpha_sum = gammaln(np.sum(self.alpha))

        if not len(alpha) == self.K:
            raise ValueError("alpha doit être un vecteur de taille K")

        if YX is None:
            self._init_XY()  # Initialisation aléatoire des assignations de cluster ligne/colonne
        else:
            self.Y, self.X = YX
        self._init_params()  # Initialisation des paramètres du modèle et variationnels

        self.lower_bound = -np.inf  # borne inférieure variationnelle

    def fit(self, max_iter=20, max_iterVEM=100, verbose=1):
        """
        Ajuste le modèle LTBM en utilisant l'algorithme VEM (Variational EM) avec une recherche gloutonne (Greedy search).

        @param max_iter: nombre maximum d'itérations de l'algorithme VEM + Greedy
        @param max_iterVEM: nombre maximum d'itérations de l'étape VEM
        @param verbose: niveau de verbosité (0: silencieux, 1: résumé d'itération, 2: détaillé VEM)
        """

        greedysucces = True
        for iter in range(max_iter):
            lb0 = self.lower_bound
            time0 = time.time()

            # Variational EM (Etape E et M)
            self.VEM(max_iterVEM, verbose=verbose > 1)
            timeVEM = time.time() - time0
            lbVEM = self.lower_bound
            deltaVEM = self.lower_bound - lb0

            if not greedysucces and deltaVEM < self.tol:
                if verbose:
                    print('\nItération {}: {:.3f} (deltaVEM: {:.4f})'.format(iter + 1, self.lower_bound, deltaVEM))
                    print('VEM: {:.2f}s\n'.format(timeVEM))
                break

            # Recherche gloutonne (Greedy search) pour améliorer Y et X
            self.greedy_search_XY()
            timeGreedy = time.time() - timeVEM - time0
            deltaGreedy = self.lower_bound - lbVEM

            if deltaGreedy > self.tol:
                greedysucces = True
            else:
                greedysucces = False

            if verbose > 0:
                print(
                    '\nItération {}: {:.3f} (deltaVEM: {:.4f}, deltaGreedy: {:.4f})'.format(iter + 1, self.lower_bound,
                                                                                            deltaVEM, deltaGreedy))
                print('VEM: {:.2f}s, Greedy: {:.2f}s\n'.format(timeVEM, timeGreedy))

            # Condition d'arrêt
            if not greedysucces and deltaVEM < self.tol:
                break

    def A_clust(self):
        """
        Retourne la matrice d'incidence avec les valeurs remplacées par les numéros de cluster (Q*X_clust + Y_clust + 1).
        """
        Y_clust = np.argmax(self.Y, axis=1)
        X_clust = np.argmax(self.X, axis=1)
        A_clust = np.zeros(self.A.shape)

        for i, j in self.T:
            A_clust[i, j] = Y_clust[i] + X_clust[j] * self.Q + 1

        return A_clust

    def _init_XY(self):
        """
        Initialise les matrices d'assignation de cluster Y (lignes) et X (colonnes) de manière aléatoire.
        """
        self.Y = np.zeros((self.A.shape[0], self.Q))
        for i in range(self.A.shape[0]):
            self.Y[i, np.random.randint(0, self.Q)] = 1

        self.X = np.zeros((self.A.shape[1], self.L))
        for j in range(self.A.shape[1]):
            self.X[j, np.random.randint(0, self.L)] = 1

    def _init_params(self):
        """
        Initialise les paramètres du modèle (rho, delta, pi, beta) et les paramètres variationnels (gamma, phi).
        """

        # Paramètres du modèle (Maximum de vraisemblance approximé)
        self._update_rho_delta()
        self._update_pi()
        self.beta = np.random.dirichlet(np.ones(self.V), self.K)  # Distribution de mots par sujet (K x V)
        self.ln_beta = np.log(self.beta)

        ### paramètres variationnels

        # paramétrise la distribution q(theta) (distribution de Dirichlet)
        self.gamma = np.ones((self.Q, self.L, self.K)) * self.alpha  # (Q x L x K)
        # E[log(theta)] = digamma(gamma) - digamma(somme(gamma))
        self.dg_gamma = digamma(self.gamma) - digamma(np.sum(self.gamma, axis=2))[:, :, np.newaxis]

        # paramétrise la distribution q(Z) (distribution multinomiale)
        # self.phi: dict ; clé: (i, j) ; valeur: liste de matrices (longueur_doc x K) des probabilités de sujet par mot.
        self.phi = {}

        for i, j in self.T:
            l = []
            # Pour chaque document dans la cellule (i, j)
            for doc in self.W[(i, j)]:
                l.append(np.ones((len(doc), self.K)) / self.K)  # Initialisation uniforme

            # Stocker dans self.phi
            self.phi[(i, j)] = l

    def VEM(self, max_iter=100, verbose=True):
        """
        Effectue les étapes variationnelles E et M (VEM).
        """

        for i in range(max_iter):
            old_lb = self.lower_bound

            # E-step: Mise à jour des paramètres variationnels (phi, gamma)
            self._update_phi()
            self._update_gamma()

            # M-step: Mise à jour des paramètres du modèle (rho, delta, pi, beta)
            self._update_rho_delta()
            self._update_pi()
            self._update_beta()

            # Calcul de la borne inférieure
            self.compute_lower_bound()

            delta = self.lower_bound - old_lb
            if verbose:
                print('Itération {}: {:.3f} (delta: {:.4f})'.format(i + 1, self.lower_bound, delta))
            if delta < self.tol:
                break

    def _update_rho_delta(self):
        """
        Mise à jour des probabilités de cluster ligne (rho) et colonne (delta).
        """
        self.rho = np.sum(self.Y, axis=0) / self.A.shape[0]  # P(Y=q)
        self.delta = np.sum(self.X, axis=0) / self.A.shape[1]  # P(X=l)

    def _update_pi(self):
        """
        Mise à jour des probabilités d'existence de lien conditionnelles (pi).
        """
        self.pi = np.zeros((self.Q, self.L))
        for q, l in np.ndindex(self.Q, self.L):
            # Indicateur si i est dans q et j est dans l
            in_clus_ql = self.Y[:, np.newaxis, q] @ self.X[np.newaxis, :, l]
            # pi_ql = Somme(A_ij * I(i in q, j in l)) / Somme(I(i in q, j in l))
            sum_A = np.sum(in_clus_ql * self.A)
            sum_clus = np.sum(in_clus_ql)
            if sum_clus > 0:
                self.pi[q, l] = sum_A / sum_clus
            else:
                self.pi[q, l] = 1e-10  # Évite la division par zéro dans le cas où le cluster est vide

    def _update_beta(self):
        """
        Mise à jour des distributions de sujets par mot (beta).
        """
        self.beta = np.zeros((self.K, self.V))  # K x V

        for i, j in self.T:
            # Pour chaque document et sa distribution de sujets variationnelle
            for doc, phi in zip(self.W[(i, j)], self.phi[(i, j)]):
                # doc: liste des indices de mots, phi: matrice (longueur_doc x K)
                for v, p in zip(doc, phi):
                    # p est le vecteur des probabilités de sujet [p(Z=k|...), k=1..K] pour le mot v
                    for k in range(self.K):
                        self.beta[k, v] += p[k]  # Accumulation de l'attribut de sujet pour le mot v

        if not np.all(self.beta > 0):
            # print('Avertissement: beta a des valeurs non positives')
            self.beta += 1e-10

        # Normalisation pour que chaque ligne de beta soit une distribution de probabilité
        self.beta = self.beta / np.sum(self.beta, axis=1)[:, np.newaxis]

        self.ln_beta = np.log(self.beta)

        # print('beta', self.beta.shape, np.sum(self.beta, axis=1))

    def _update_phi(self):
        """
        Mise à jour des probabilités variationnelles de sujet par mot (phi).
        """
        for i, j in self.T:
            # Récupérer les clusters assignés
            q = np.where(self.Y[i, :] == 1)[0][0]
            l = np.where(self.X[j, :] == 1)[0][0]

            # Pré-calculer les termes exponentiels basés sur E[log(theta)]
            exp_comp_k = np.exp(self.dg_gamma[q, l, :])  # Vecteur de taille K

            # Mise à jour de phi pour chaque document dans la cellule (i, j)
            for doc_idx, (doc, phi) in enumerate(zip(self.W[(i, j)], self.phi[(i, j)])):

                # phi est une matrice (longueur_doc x K)
                for word_idx, v in enumerate(doc):
                    # p[k] = exp(E[log(theta_qlk)]) * beta_kv
                    # Utiliser la multiplication de numpy pour l'efficacité
                    phi[word_idx, :] = self.beta[:, v] * exp_comp_k  # (K,) * (K,) -> (K,)

                # Normalisation pour que la somme sur K soit 1 pour chaque mot
                if not np.all(phi > 0):
                    # print('Avertissement: phi a des valeurs non positives')
                    phi += 1e-10  # Assurer des valeurs positives pour la normalisation

                # Normalisation
                sum_phi = np.sum(phi, axis=1)[:, np.newaxis]
                phi /= sum_phi

                # L'objet `phi` est modifié en place, pas besoin de réassigner

    def _update_gamma(self):
        """
        Mise à jour des paramètres variationnels de Dirichlet (gamma).
        """

        # Initialisation: gamma_qlk = alpha_k
        self.gamma = np.ones((self.Q, self.L, self.K)) * self.alpha

        # Accumulation de la somme des phi (E[Z])
        for i, j in self.T:
            q = np.where(self.Y[i, :] == 1)[0][0]
            l = np.where(self.X[j, :] == 1)[0][0]

            for doc_idx in range(len(self.W[(i, j)])):
                # Ajouter la somme de la matrice phi_d (somme sur les mots) à gamma_ql
                self.gamma[q, l, :] += np.sum(self.phi[(i, j)][doc_idx], axis=0)

        # Mise à jour de E[log(theta)]
        self.dg_gamma = digamma(self.gamma) - digamma(np.sum(self.gamma, axis=2))[:, :, np.newaxis]
        # Mise à jour des termes de la fonction gamma log
        self.gln_gamma = gammaln(self.gamma)
        self.gln_gamma_sum = gammaln(np.sum(self.gamma, axis=2))

    def compute_lower_bound(self):
        """
        Calcule la borne inférieure variationnelle (Evidence Lower Bound - ELBO).
        """
        lb = 0

        #### L(q(.) | A, Y, X, beta) (se référer à l'Annexe C) ###
        for i, j in self.T:
            q = np.where(self.Y[i, :] == 1)[0][0]
            l = np.where(self.X[j, :] == 1)[0][0]

            for doc, phi in zip(self.W[(i, j)], self.phi[(i, j)]):
                # phi: matrice (longueur_doc x K), doc: liste d'indices de mots
                for v, p in zip(doc, phi):  # v est l'indice du mot, p est le vecteur de probabilité de sujet (taille K)
                    for k in range(self.K):
                        # p[k] est E[Z^dnk]

                        # 1er terme: E[log p(W|Z, beta)] = E[Z^dnk] * log(beta_kv)
                        lb += p[k] * self.ln_beta[k, v]

                        # 2ème terme: E[log p(Z|Y, X, theta)] = E[Z^dnk] * E[log(theta_qlk)]
                        lb += p[k] * self.dg_gamma[q, l, k]

                        # 4ème terme: -E[log q(Z)] = - E[Z^dnk] * log(phi_dnk)
                        # Pour éviter log(0), nous avons ajouté un petit epsilon lors de la mise à jour de phi,
                        # mais il est plus sûr d'utiliser np.log(p[k]) directement car p[k] est déjà positif.
                        if p[k] > 0:
                            lb -= p[k] * np.log(p[k])

        for q, l in np.ndindex(self.Q, self.L):
            # 3ème terme: E[log p(theta|alpha)]
            # Γ(Σαk) - ΣΓ(αk) + Σ(αk-1) * E[log θqlk]
            lb += self.gln_alpha_sum - np.sum(self.gln_alpha) + np.sum((self.alpha - 1) * self.dg_gamma[q, l, :])

            # 5ème terme: -E[log q(theta)]
            # - (log Γ(Σγ) - Σlog Γ(γ) + Σ(γk-1) * E[log θqlk])
            lb -= self.gln_gamma_sum[q, l] - np.sum(self.gln_gamma[q, l, :]) + np.sum(
                (self.gamma[q, l, :] - 1) * self.dg_gamma[q, l, :])

        #### p(A, Y, X | pi, rho, delta) (éq 15) ###
        # Cette partie est log p(A, Y, X|pi, rho, delta)
        for i, j in self.T:
            q = np.where(self.Y[i, :] == 1)[0][0]
            l = np.where(self.X[j, :] == 1)[0][0]

            # Aij=1, donc on calcule seulement log(pi_ql * rho_q * delta_l)
            lb += np.log(self.pi[q, l] * self.rho[q] * self.delta[l])

        # print('lb', lb)

        self.lower_bound = lb

    def greedy_search_XY(self):
        """
        Effectue une recherche gloutonne pour améliorer les assignations de cluster Y (lignes) et X (colonnes)
        en maximisant la borne inférieure.
        """

        # Recherche gloutonne pour Y (lignes)
        qs_num = np.sum(self.Y, axis=0)  # Nombre d'éléments dans chaque cluster ligne
        for i in range(self.A.shape[0]):
            q = np.argmax(self.Y[i, :])  # Cluster actuel de la ligne i

            # On ne change de cluster que s'il y a plus d'un élément dans le cluster actuel
            if qs_num[q] > 1:
                current_lb = self.lower_bound
                best_lb = current_lb
                q_final = q  # Meilleur cluster trouvé

                self.Y[i, q] = 0  # Retirer i de son cluster actuel

                # Re-calculer les paramètres (pi, rho, delta) pour que compute_lower_bound fonctionne correctement
                self._update_rho_delta()
                self._update_pi()

                for q_try in range(self.Q):
                    if q_try == q: continue

                    self.Y[i, q_try] = 1  # Essayer le cluster q_try
                    # Mise à jour des paramètres du modèle nécessaires
                    self._update_rho_delta()
                    self._update_pi()
                    self.compute_lower_bound()  # Calcul de la borne inférieure pour l'essai

                    if self.lower_bound > best_lb:
                        q_final = q_try
                        best_lb = self.lower_bound

                    self.Y[i, q_try] = 0  # Retirer i du cluster q_try

                # Rétablir i dans le meilleur cluster trouvé
                self.Y[i, q_final] = 1
                self._update_rho_delta()
                self._update_pi()
                self.lower_bound = best_lb

                # Mise à jour du nombre d'éléments dans les clusters
                qs_num[q] -= 1
                qs_num[q_final] += 1

        # Recherche gloutonne pour X (colonnes)
        ls_num = np.sum(self.X, axis=0)  # Nombre d'éléments dans chaque cluster colonne
        for j in range(self.A.shape[1]):
            l = np.argmax(self.X[j, :])  # Cluster actuel de la colonne j

            if ls_num[l] > 1:
                current_lb = self.lower_bound
                best_lb = current_lb
                l_final = l

                self.X[j, l] = 0

                # Re-calculer les paramètres (pi, rho, delta)
                self._update_rho_delta()
                self._update_pi()

                for l_try in range(self.L):
                    if l_try == l: continue

                    self.X[j, l_try] = 1
                    # Mise à jour des paramètres du modèle nécessaires
                    self._update_rho_delta()
                    self._update_pi()
                    self.compute_lower_bound()

                    if self.lower_bound > best_lb:
                        l_final = l_try
                        best_lb = self.lower_bound

                    self.X[j, l_try] = 0

                # Rétablir j dans le meilleur cluster trouvé
                self.X[j, l_final] = 1
                self._update_rho_delta()
                self._update_pi()
                self.lower_bound = best_lb

                # Mise à jour du nombre d'éléments dans les clusters
                ls_num[l] -= 1
                ls_num[l_final] += 1