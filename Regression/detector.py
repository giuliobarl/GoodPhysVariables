import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import combinations
from scipy import linalg
from scipy.optimize import least_squares
import tqdm

from Regression.dataset import InvarianceDataset
from Regression.model import InvarianceModel

@tf.function
def get_gradients(model, x, indices):
    """
    A compiled function to efficiently compute gradients.
    """
    with tf.GradientTape() as tape:
        tape.watch(x)
        predictions = model(x)
    
    # Get the gradient of the prediction with respect to the input tensor
    grads = tape.gradient(predictions, x)
    
    # Return the specific gradients for the feature pair
    return [grads[0, idx] for idx in indices]

def pair_objective(
        beta: list[float],
        model: tf.keras.Model, # Pass the actual Keras model
        x0_np: np.ndarray,
        idxs: list[int],
        univec_fn,
        mode: str,
        eps_norm: float = 1e-12
        ):
    """
    Objective for pairs invariance.

    Parameters
    ----------
    beta : list[float]
        Initial guesses [a, b].
    model : InvarianceModel
        Trained NN model.
    x0_np : np.ndarray
        Evaluation point.
    idxs : list[int]
        Indices of features (idx1, idx2).
    univec_fn : callable
        Function to build null-space basis.

    Returns
    -------
    np.ndarray
        Residuals enforcing invariance conditions.
    """
    a0, b0 = beta
    idx1, idx2 = idxs
    
    # Convert input to a tensor ONCE
    x0_tf = tf.constant(x0_np.reshape(1, -1), dtype=tf.float32)

    # Call the compiled gradient function
    grads_tf = get_gradients(model, x0_tf, [idx1, idx2])    
    grads = np.array([g.numpy() for g in grads_tf], dtype=float)

    norm = np.linalg.norm(grads)

    if not np.isfinite(norm) or norm < eps_norm:
        # Return a large residual if gradient is unstable
        return np.array([1e3, a0**2 + b0**2 - 1.0], dtype=float)
    
    grads /= norm

    un = univec_fn(a0, b0, x0_np, idxs, mode)
    if un.ndim == 2:
        un = un[:, 0]

    return np.array([
        np.dot(grads, un),
        a0**2 + b0**2 - 1.0
        ], dtype=float)

def triplet_objective(
        beta: list[float],
        model: tf.keras.Model,
        x0_np: np.ndarray,
        idxs: list[int],
        univec_fn,
        mode: str,
        eps_norm: float = 1e-12
        ):
    """
    Objective for triplets invariance.

    Parameters
    ----------
    beta : list[float]
        Initial guesses [a, b, c].
    model : InvarianceModel
        Trained NN model.
    x0_np : np.ndarray
        Evaluation point.
    idxs : list[int]
        Indices of features (idx1, idx2, idx3).
    univec_fn : callable
        Function to build null-space basis.

    Returns
    -------
    np.ndarray
        Residuals enforcing invariance conditions.
    """
    a0, b0, c0 = beta
    idx1, idx2, idx3 = idxs

    x0_tf = tf.constant(x0_np.reshape(1, -1), dtype=tf.float32)

    # gradients for three features
    grads_tf = get_gradients(model, x0_tf, [idx1, idx2, idx3])
    grads = np.array([g.numpy() for g in grads_tf], dtype=float)

    norm = np.linalg.norm(grads)
    if not np.isfinite(norm) or norm < eps_norm:
        return np.array([1e3, 1e3, a0**2 + b0**2 + c0**2 - 1.0], dtype=float)
    grads /= norm

    un = univec_fn(a0, b0, c0, x0_np, idxs, mode)
    un1, un2 = un[:, 0], un[:, 1]  # two independent orth vectors

    return np.array([
        np.dot(grads, un1),
        np.dot(grads, un2),
        a0**2 + b0**2 + c0**2 - 1.0
        ], dtype=float)

def coupled_pairs_objective(
        beta: list[float],
        model: tf.keras.Model,
        x0_np: np.ndarray,
        idxs: list[int],
        univec_fn,
        mode: str,
        eps_norm: float = 1e-12
        ):
    """
    Objective for coupled pairs invariance.

    Parameters
    ----------
    beta : list[float]
        Initial guesses [a, b, c, d].
    model : InvarianceModel
        Trained NN model.
    x0_np : np.ndarray
        Evaluation point.
    idxs : list[int]
        Indices of features (idx1, idx2, idx3, idx4).
    univec_fn : callable
        Function to build null-space basis.

    Returns
    -------
    np.ndarray
        Residuals enforcing invariance conditions.
    """
    a, b, c, d = beta
    idx1, idx2, idx3, idx4 = idxs

    # gradient
    x0_tf = tf.constant(x0_np.reshape(1, -1), dtype=tf.float32)

    # gradients
    grads_tf = get_gradients(model, x0_tf, [idx1, idx2, idx3, idx4])
    grads = np.array([grads_tf[0], grads_tf[1], grads_tf[3]], dtype=float)

    norm = np.linalg.norm(grads)
    if not np.isfinite(norm) or norm < eps_norm:
        return np.array([1e3, a**2 + b**2 + c**2 + d**2 - 1.0], dtype=float)
    grads /= norm

    # null-space vector
    un = univec_fn(a, b, c, d, x0_np, idxs)
    if un.ndim == 2:
        un = un[:, 0]  # first vector, since we only need 1D invariance

    return np.array([
        np.dot(grads, un),
        a**2 + b**2 -1.0,
        c**2 + d**2 - 1.0
        ], dtype=float)



class InvarianceDetector:
    def __init__(self):
        """
        Parameters
        ----------
        model : InvarianceModel
            InvarianceModel object.
        feature_names : list
            Names of input features in the same order as model inputs.
        """
        self.feature_pairs = []
        self.feature_triplets = []
        self.coupled_pairs = []
        self.A_mat = []
        self.B_mat = []
        self.C_mat = []
        self.D_mat = []

    def _univec_pair(self, a, b, x0, idxs, mode='product'):
        idx1, idx2 = idxs
        B = np.zeros((1, 2))
        if mode == 'product':
            B[0, 0] = a / x0[idx1]
            B[0, 1] = b / x0[idx2]
        elif mode == 'sum':
            B[0, 0] = a
            B[0, 1] = b
        else:
            raise ValueError(f"Mode must be one of ['product', 'sum'].")
        return linalg.null_space(B)
    
    def _univec_triplet(self, a, b, c, x0, idxs, mode='product'):
        idx1, idx2, idx3 = idxs
        B = np.zeros((1, 3))
        if mode == 'product':
            B[0, 0] = a / x0[idx1]
            B[0, 1] = b / x0[idx2]
            B[0, 2] = c / x0[idx3]
        elif mode == 'sum':
            B[0, 0] = a
            B[0, 1] = b
            B[0, 2] = c
        else:
            raise ValueError(f"Mode must be one of ['product', 'sum'].")
        return linalg.null_space(B)  # shape (3,2), two orthogonal basis vectors
    
    def _univec_coupled_pairs(self, a, b, c, d, x0, idxs):
        """
        Compute the null-space vector for a coupled-pairs invariance check.

        Parameters
        ----------
        a, b, c, d : float
            Exponents for the two couples.
        x0 : np.ndarray
            Feature values at which to evaluate.
        idxs : list[int]
            Indices of the three involved features.
            Example: couples = (f1, f2), (f1, f3).

        Returns
        -------
        un : np.ndarray
            A null-space basis vector (shape (3,) or (3,1)).
        """

        idx1, idx2, idx3, idx4 = idxs
        B = np.zeros((2, 3))

        if idx1 == idx3:
            # (x1, x2), (x1, x3)
            B[0, 0] = a / x0[idx1]
            B[0, 1] = b / x0[idx2]
            B[1, 0] = c / x0[idx3]
            B[1, 2] = d / x0[idx4]

        elif idx2 == idx3:
            # (x1, x2), (x2, x3)
            B[0, 0] = a / x0[idx1]
            B[0, 1] = b / x0[idx2]
            B[1, 1] = c / x0[idx3]
            B[1, 2] = d / x0[idx4]

        un = linalg.null_space(B)

        if un.ndim > 1:
            un = un[:, 0]

        return un

    def find_invariant_pairs(
            self,
            dataset: InvarianceDataset,
            model: InvarianceModel,
            guess: tuple[float],
            mode: str = "product",
            N: int = 20,
            repeat: int = 3,
            verbose: bool = False
            ):
        """
        Detect invariant exponent pairs (alpha, beta) for all feature pairs.

        Parameters
        ----------
        dataset: InvarianceDataset
            Dataset with all input features (must match training order).
        model: InvarianceModel
            Trained DNN model.
        a_guess : float
            Initial guess for the first exponent.
        b_guess : float
            Initial guess for the second exponent.
        mode: str
            Type of functional relationship between the variables in the searched modes.
        N : int
            Number of random points per repeat.
        repeat : int
            Number of repeat optimizations per pair.
        verbose : bool
            Whether to show progress bars.

        Returns
        -------
        pd.DataFrame
            Columns: mean a, std a, mean b, std b for each feature pair.
        """
        data_df = dataset.df
        results_a, results_b = [], []
        feature_pairs = list(combinations(dataset.feature_names, 2))
        self.feature_pairs = feature_pairs

        feat_means = data_df[dataset.feature_names].describe().loc['mean'].to_dict()
        feat_mins = data_df[dataset.feature_names].min().to_dict()
        feat_maxs = data_df[dataset.feature_names].max().to_dict()

        for pair in tqdm.tqdm(feature_pairs):
            a_vals = guess[0] * np.ones(N)
            b_vals = guess[1] * np.ones(N)

            for _ in range(repeat):
                beta_guess = np.array([np.mean(a_vals), np.mean(b_vals)])
                a_vals = np.ones(N)
                b_vals = np.ones(N)

                for k in range(N):
                    # --- sample a deterministic x0 for this iteration ---
                    tx = feat_means.copy()
                    for var in pair:
                        vmin, vmax = feat_mins[var], feat_maxs[var]
                        tx[var] = vmin + np.random.uniform(0.0, 1.0) * (vmax - vmin)

                    x0_np = np.array([tx[f] for f in dataset.feature_names], dtype=float)

                    # Indexes of the two features
                    idxs = list([dataset.feature_names.index(v) for v in pair])

                    keras_model = model.model

                    res = least_squares(
                        pair_objective,
                        beta_guess,
                        args=(keras_model, x0_np, idxs, self._univec_pair, mode),
                        method='trf',
                        ftol=1e-12,
                        max_nfev=50,
                        verbose=verbose
                        )
                    
                    res.x = res.x / np.linalg.norm(res.x)
                    a_vals[k], b_vals[k] = res.x

            self.A_mat.append(a_vals)
            self.B_mat.append(b_vals)

            results_a.append((np.mean(a_vals), np.std(a_vals)))
            results_b.append((np.mean(b_vals), np.std(b_vals)))

        # Build DataFrame
        df_results = pd.DataFrame({
            'feature_pair': feature_pairs,
            'mean_a': [m for m, s in results_a],
            'std_a': [s for m, s in results_a],
            'mean_b': [m for m, s in results_b],
            'std_b': [s for m, s in results_b]
        })
        return df_results
    
    def find_invariant_triplets(
            self,
            dataset: InvarianceDataset,
            model: InvarianceModel,
            guess: tuple[float],
            mode: str = "product",
            N: int = 20,
            repeat: int = 3,
            verbose: bool = False
        ):
        """
        Detect invariant exponent/coefficient triplets (alpha, beta, gamma) for all feature triplets.

        Parameters
        ----------
        dataset: InvarianceDataset
            Dataset with all input features (must match training order).
        model: InvarianceModel
            Trained DNN model.
        a_guess : float
            Initial guess for the first exponent/coefficient.
        b_guess : float
            Initial guess for the second exponent/coefficient.
        c_guess : float
            Initial guess for the second exponent/coefficient.
        mode: str
            Type of functional relationship between the variables in the searched modes.
        N : int
            Number of random points per repeat.
        repeat : int
            Number of repeat optimizations per triplet.
        verbose : bool
            Whether to show progress bars.

        Returns
        -------
        pd.DataFrame
            Columns: mean a, std a, mean b, std b, mean c, std c for each feature triplet.
        """
        data_df = dataset.df
        results_a, results_b, results_c = [], [], []
        feature_triplets = list(combinations(dataset.feature_names, 3))
        self.feature_triplets = feature_triplets

        feat_means = data_df[dataset.feature_names].describe().loc['mean'].to_dict()
        feat_mins = data_df[dataset.feature_names].min().to_dict()
        feat_maxs = data_df[dataset.feature_names].max().to_dict()

        for triplet in tqdm.tqdm(feature_triplets):
            a_vals = guess[0] * np.ones(N)
            b_vals = guess[1] * np.ones(N)
            c_vals = guess[2] * np.ones(N)

            for _ in range(repeat):
                beta_guess = np.array([np.mean(a_vals), np.mean(b_vals), np.mean(c_vals)])
                a_vals = np.ones(N)
                b_vals = np.ones(N)
                c_vals = np.ones(N)

                for k in range(N):
                    tx = feat_means.copy()
                    for var in triplet:
                        vmin, vmax = feat_mins[var], feat_maxs[var]
                        tx[var] = vmin + np.random.uniform(0.0, 1.0) * (vmax - vmin)

                    x0_np = np.array([tx[f] for f in dataset.feature_names], dtype=float)
                    idxs = list([dataset.feature_names.index(v) for v in triplet])

                    keras_model = model.model

                    res = least_squares(
                        triplet_objective,
                        beta_guess,
                        args=(keras_model, x0_np, idxs, self._univec_triplet, mode),
                        method='trf',
                        ftol=1e-12,
                        max_nfev=50,
                        verbose=verbose
                    )

                    res.x = res.x / np.linalg.norm(res.x)
                    a_vals[k], b_vals[k], c_vals[k] = res.x

            self.A_mat.append(a_vals)
            self.B_mat.append(b_vals)
            self.C_mat.append(c_vals)

            results_a.append((np.mean(a_vals), np.std(a_vals)))
            results_b.append((np.mean(b_vals), np.std(b_vals)))
            results_c.append((np.mean(c_vals), np.std(c_vals)))

        df_results = pd.DataFrame({
            'feature_triplet': feature_triplets,
            'mean_a': [m for m, s in results_a],
            'std_a': [s for m, s in results_a],
            'mean_b': [m for m, s in results_b],
            'std_b': [s for m, s in results_b],
            'mean_c': [m for m, s in results_c],
            'std_c': [s for m, s in results_c]
        })
        return df_results
    
    def find_invariant_coupled_pairs(
            self,
            dataset: InvarianceDataset,
            model: InvarianceModel,
            guess: tuple[float],
            mode: str = "product",
            N: int = 20,
            repeat: int = 3,
            verbose: bool = False
            ):
        """
        Detect invariant exponent pairs (alpha, beta) for all feature pairs.

        Parameters
        ----------
        dataset: InvarianceDataset
            Dataset with all input features (must match training order).
        model: InvarianceModel
            Trained DNN model.
        a_guess : float
            Initial guess for the first exponent.
        b_guess : float
            Initial guess for the second exponent.
        mode: str
            Type of functional relationship between the variables in the searched modes.
        N : int
            Number of random points per repeat.
        repeat : int
            Number of repeat optimizations per pair.
        verbose : bool
            Whether to show progress bars.

        Returns
        -------
        pd.DataFrame
            Columns: mean a, std a, mean b, std b for each feature pair.
        """
        data_df = dataset.df
        results_a, results_b, results_c, results_d = [], [], [], []

        l = list(combinations(dataset.feature_names, 2))
        ll = list(combinations(l, 2))
        coupled_pairs = []
        for i in range(len(ll)):
            if ll[i][1][0] in ll[i][0]:
                coupled_pairs.append(ll[i])
                
            elif ll[i][1][1] in ll[i][0]:
                coupled_pairs.append([ll[i][0], ll[i][1][::-1]])

        self.coupled_pairs = coupled_pairs

        feat_means = data_df[dataset.feature_names].describe().loc['mean'].to_dict()
        feat_mins = data_df[dataset.feature_names].min().to_dict()
        feat_maxs = data_df[dataset.feature_names].max().to_dict()

        for coupled_pair in tqdm.tqdm(coupled_pairs):
            a_vals = guess[0] * np.ones(N)
            b_vals = guess[1] * np.ones(N)
            c_vals = guess[2] * np.ones(N)
            d_vals = guess[3] * np.ones(N)

            for _ in range(repeat):
                beta_guess = np.array([
                    np.mean(a_vals),
                    np.mean(b_vals),
                    np.mean(c_vals),
                    np.mean(d_vals)
                    ])
                a_vals = np.ones(N)
                b_vals = np.ones(N)
                c_vals = np.ones(N)
                d_vals = np.ones(N)

                for k in range(N):
                    # --- sample a deterministic x0 for this iteration ---
                    tx = feat_means.copy()
                    for pair in coupled_pair:
                        for var in pair:
                            vmin, vmax = feat_mins[var], feat_maxs[var]
                            tx[var] = vmin + np.random.uniform(0.0, 1.0) * (vmax - vmin)

                    x0_np = np.array([tx[f] for f in dataset.feature_names], dtype=float)

                    # Indexes of the two features        
                    idxs = [dataset.feature_names.index(v) for pair in coupled_pair for v in pair]

                    keras_model = model.model

                    res = least_squares(
                        coupled_pairs_objective,
                        beta_guess,
                        args=(keras_model, x0_np, idxs, self._univec_coupled_pairs, mode),
                        method='trf',
                        ftol=1e-12,
                        max_nfev=50,
                        verbose=verbose
                        )
                    
                    a_vals[k], b_vals[k], c_vals[k], d_vals[k] = res.x

            self.A_mat.append(a_vals)
            self.B_mat.append(b_vals)
            self.C_mat.append(c_vals)
            self.D_mat.append(d_vals)

            results_a.append((np.mean(a_vals), np.std(a_vals)))
            results_b.append((np.mean(b_vals), np.std(b_vals)))
            results_c.append((np.mean(c_vals), np.std(c_vals)))
            results_d.append((np.mean(d_vals), np.std(d_vals)))

        # Build DataFrame
        df_results = pd.DataFrame({
            'coupled_pair': coupled_pairs,
            'mean_a': [m for m, s in results_a],
            'std_a': [s for m, s in results_a],
            'mean_b': [m for m, s in results_b],
            'std_b': [s for m, s in results_b],
            'mean_c': [m for m, s in results_c],
            'std_c': [s for m, s in results_c],
            'mean_d': [m for m, s in results_d],
            'std_d': [s for m, s in results_d]
        })
        return df_results

    def plot_results(self, group, **kwargs):
        if group == "pair":
            for ind in range(len(self.feature_pairs)):
                plt.figure(**kwargs)
                plt.scatter(np.arange(len(self.A_mat[ind])), self.A_mat[ind], label=r'$\alpha$')
                plt.scatter(np.arange(len(self.B_mat[ind])), self.B_mat[ind], label=r'$\beta$')
                plt.title(self.feature_pairs[ind])
                plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
                plt.ylim(-1, 1)
                plt.show()
        elif group == "triplet":
            for ind in range(len(self.feature_triplets)):
                plt.figure(**kwargs)
                plt.scatter(np.arange(len(self.A_mat[ind])), self.A_mat[ind], label=r'$\alpha$')
                plt.scatter(np.arange(len(self.B_mat[ind])), self.B_mat[ind], label=r'$\beta$')
                plt.scatter(np.arange(len(self.C_mat[ind])), self.C_mat[ind], label=r'$\gamma$')
                plt.title(self.feature_triplets[ind])
                plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
                plt.ylim(-1, 1)
                plt.show()
        else:
            raise ValueError(f"Group must be one of ['pair', 'triplet'].")