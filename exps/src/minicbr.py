# PRR Definition
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import precision_recall_fscore_support as prf
from sklearn.metrics import matthews_corrcoef as mcc

class MiniCBR:
    """
        A custom recovery class tailored to work with patient recrods for which
        similarity must be assessed taking multiple features of various type into account.
    """
    def __init__(self, df_X, df_y, feature_map={}):
        """
            Params:
                df_X: Dataframe of entries. Collection of records from which to recover the most similar ones. Essentially, the case base
                    NOTE: Relevant features needs to be binarized beforehand, otherwise recovery wont work!
                df_Y: Dataframe of solutions for each entry in df_X. NOTE TODO: For now, only bool types work ok.
                feature_map: Dictionary that maps prefixes of case features with 2-tuples of kind:
                    (metric, normalize, weight), where:
                        metric identifies the strategy to use to compute the distance
                        normalize is a boolean value identifying wherther to normalize distances for each entry
                        weight is the ponderation after resolving distances for that group of features
        """
        self.cb_X = df_X
        
        #self.cb_y = df_y    # Only used in fit
        self.cb_y  = np.array(df_y, dtype=int) # Cast to int on init. Assumes working with booleans
        
        self.feature_map = feature_map
        self.n_entries = len(self.cb_X)

        # Resolves df_X_data. Collection of features that will be iterated in find() when resolving distances from various attributes
        #   This generates a dictionary of tuples --> prefix: (column_names, data_as_array)
        self.cb_arr_features = {}
        if self.cb_X is not None:
            cb_arr_features = {}

            # Gets data from cb_X belonging to each prefix
            for prefix in self.feature_map.keys():

                # Resolves actual columns with prefix
                prefix_cnames = [c for c in self.cb_X if c.startswith(prefix)]

                # Feature prepro (select, cast and fix). Gets features as numpy arrays
                f_arr_X = self._feature_prepro(self.cb_X, prefix_cnames)

                # Saves processed features
                cb_arr_features[prefix] = (prefix_cnames, f_arr_X)
            self.cb_arr_features = cb_arr_features

    def fit(self, df_train_X, df_train_y, k=1, delta_weight=.1, score_metric='mcc', weighted_voting=True, verbose=False):
        """
            Explores feature weights and keeps the optimal combination to resolve the features given by target_columns.
            The objetive is to find the solutions of a case by aggregating the known solution of its closest neighbours.
            The rationale behind this fitting method is that a good feature weight distribution results in finding better neighbours,
                thus we can explore the weight space to approximate this optimality.

            Params:
                df_train_X: Dataframe. Data to use for fitting
                df_train_y: Dataframe. Solutions for df_train_X
                target_columns: List of column names in df_train_X that contain the expected solutions to the case
                    NOTE: For now, only boolean types are considered for the target columns. Other types will behave differently.
                k: Number of neighbours to take into account when resolving a solution from them
                delta_weight: float. Increment to take into account when finding optimal weights. The smaller, the finer.
                score_metric: One of: {uf1, mf1, mcc}. Evaluation metric to consider, respectively, microf1, macrof1, matthews correlation coefficient
                weighted_voting: Bool. If set, neighbour distance is taken into account when aggregating solutions.
                NOTE: k, score_metric and weighted_voting are only taken into account for fitting. A different k value can be later specified when calling find(),
                    and weighted_voting is irrelevant after fitting, as aggregating the solutions is not actually carried out by this class,
                    rather is only used as a tool to measure the quality of the retrieved cases by looking at their combined solutions,
                    this is specially true when k=1, since we'll then be comparing the outcomes between the target and retrieved case directly.
        """        
        cb_y = self.cb_y
        train_y = np.array(df_train_y, dtype=int)   # Cast to int. Assumes bools

        # First we find the distance matrix for each feature group between the case base and df_train_X
        f_dmatrix = {}    # Distance matrices for each feature group
        dmatrix_shape = (len(df_train_X), len(self.cb_X))
        def_score = 'mcc'
        for f_prefix, f_opts in self.feature_map.items():
            if verbose: print(f'>> Computing distance matrix for feature group {f_prefix}...')

            f_metric, f_norm, _ = f_opts

            # Select appropriate columns for the feature group in df (assumes binarized already)
            f_cnames, f_arr_cb = self.cb_arr_features[f_prefix]

            # Sorts out data from train
            f_arr_x = self._feature_prepro(df_train_X, f_cnames)
            
            # Compute distance (batch; f_dist.shape = (len(train), len(cb))),
            #   ie, each row consist of the distance for a given case in train to each of the cases in cb
            f_dist = scipy.spatial.distance.cdist(f_arr_x, f_arr_cb, metric=f_metric)

            if f_norm:
                # Distance normalization
                f_dist = f_dist / np.linalg.norm(f_dist, axis=1)[:, np.newaxis]

            f_dmatrix[f_prefix] = f_dist
        if verbose: print('\n')

        # Fixed weight combos
        # emb, age, gender, admission_type, insurance, marital_status, race, drg_mortality, drg_code
        #weight_combos = [
        #    (0.6, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05),  # Mostly embeddings
        #    (0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11),  # Same weights
        #    (0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0),          # Emb + mortality
        #    (0.33, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.33, 0.33),          # Emb + mortality + drg code
        #    (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)          # Only embeddings
        #]


        # Getting feature weight combinations via recursion
        def __recursive_weights(weights_list, n_remaining_features, known_weights):
            remaining_weight = 1.0 - sum(known_weights)
            if n_remaining_features == 1:
                # Last feature weight to assign is the required to sum 1.0 with the already known ones
                return [known_weights + [float(remaining_weight)]]
            
            else:
                # Keep resolved weight combinations
                combos = []

                # Iterates weights
                for w in weights_list:
                    # Only make recursive call if there are weights to assign
                    if w <= remaining_weight:
                        # Update weight combo
                        known_weights_i = known_weights + [float(w)]

                        # Recursive call to get the resulting combinations of the current partial combo
                        combos += __recursive_weights(weights_list, n_remaining_features-1, known_weights_i)
                    else:
                        break

            return combos
            
        __n_wfeatures = len(self.feature_map.keys())
        if verbose: print(f'>> Generating weight combos for {__n_wfeatures} features with a delta of {delta_weight}...')
        __weights_list = np.linspace(0.0, 1.0, int((1.0 // delta_weight) + 1))
        weight_combos = __recursive_weights(__weights_list, __n_wfeatures, [])
        if verbose: print(f'<< Generated {len(weight_combos)} weight combos')

        f_groups_cnames = list(self.feature_map.keys())   # Sorted list of feature groups
        best_combo = weight_combos[0]
        best_score = 0
        for i, w_combo in enumerate(weight_combos):
            # Iterates each weight combo
            if verbose: print(f'>> Checking combo {i+1} out of {len(weight_combos)}...', end='\r')

            # Apply weights to each matrix in dmatrix and sums cell-wise
            weighted_dmatrix = np.zeros(shape=dmatrix_shape)
            for f_name, f_weight in zip(f_groups_cnames, w_combo):
                weighted_dmatrix += f_dmatrix[f_name] * f_weight

            q_sols = []
            for i, q_dist in enumerate(weighted_dmatrix):
                # Select the k nearest neighours for each train query using the weighted distance matrix
                nearest_cb_idx = np.argsort(q_dist)[:k]

                # Retrieves the solutions from the neighbours
                nearest_sols = np.concatenate([cb_y[ni] for ni in nearest_cb_idx], axis=0)
                
                # Aggregate by majority voting as specified
                # Applies neighbour weight if required
                # NOTE TODO: This might only work for 1d target columns
                if weighted_voting:
                    # NOTE: (!) Distances must be normalized for this to work properly
                    nearest_sols = nearest_sols * (1 - q_dist[nearest_cb_idx])

                    # Then round to nearest integer
                    nearest_sols = np.round(nearest_sols)

                # Cast to bool
                nearest_sols = nearest_sols.astype(dtype=bool)

                # Count values
                unique, counts = np.unique(nearest_sols, return_counts=True)

                # Keep the most popular solution among neighbours
                q_sol = unique[np.argsort(counts)[-1]]
                q_sols.append(q_sol)     

            # Evaluate solution with specified metric
            if score_metric not in ['mcc', 'uf1', 'mf1']:
                print(f'\n(!) Score {score_metric} unrecognize. Defaulting to {def_score}...')
                score_metric = def_score
            if score_metric == 'mcc':
                score = mcc(train_y, q_sols)

            elif score_metric == 'uf1':
                score = prf(train_y, q_sols, average='micro')[2]

            elif score_metric == 'mf1':
                score = prf(train_y, q_sols, average='macro')[2]

            else:
                raise NotImplementedError

            # Update best combo
            if score > best_score:
                best_combo = w_combo
                best_score = score

                print(f'\n(!) New best combo found: {w_combo} (score: {score})')
        if verbose: print('\n')

        # Keep best combo found as adjusted weights by updating self.feature_map accordingly
        for f_name, f_adj_weight in zip(f_groups_cnames, best_combo):
            self.feature_map[f_name] = (self.feature_map[f_name][0], self.feature_map[f_name][1], f_adj_weight)

        print('Fitting ok!')
        print(f'>> Optimal feature map weight combination:')
        print('{' + '\n'.join('{!r}: {!r},'.format(k, v) for k, v in self.feature_map.items()) + '}')
        return

    def _feature_prepro(self, df, feature_cnames):
        """
            Helper method to process feature groups in a dataframe. Selects columns, cast and fixes dimensions for a given feature group.
            Params:
                df: DataFrame containing features to process
                feature_cnames: List of column names related to the specific feature group to process
        """
        # We are expecting dataframes
        assert isinstance(df, pd.DataFrame)

        # Gets apropriate columns for a given feature group in df
        # NOTE: If len(feature_cnames) > 1, "feats" is a dataframe with the relevant feature columns, otherwise, is a list of values
        #   We are doing this to allow wrapping whole numpy tensors in df cells, which is mighty useful when using sentence embeddings as part of the case definition
        #   Without tolist(), the resulting array after casting would be of shape (n_entries, 1) instead of (n_entries, n_emb_features)
        feats = df[feature_cnames] if len(feature_cnames) > 1 else df[feature_cnames[0]].tolist()

        # We want features in tensor form to measure distance
        arr_feats = np.array(feats)

        # Fix 0-dimensional data
        arr_feats = arr_feats[:, np.newaxis] if len(arr_feats.shape) == 1 else arr_feats
        
        # Convert boolean types to int
        arr_feats = arr_feats.astype(dtype=int) if isinstance(arr_feats[0][0], (bool, np.bool_)) else arr_feats
        return arr_feats

    def find(self, df_x, k=2, verbose=False):
        """
            Retrieves neighbours to a set of queries
            Params:
                df_x: Dataframe of queries
                k: Number of neighbours to retrieve

            Returns a tuple of (distances, idxs) of shape distances.shape = idxs.shape = (n_queries, k)
        """

        # Keep distances and indices of k nearest neighbours for each query
        nearest_distances = []
        nearest_idx = []

        n_entries = len(df_x)

        # Computes distance from individual entries
        for i in range(n_entries):
            if verbose: print(f'>> Finding neighbours for query {i} out of {n_entries}...', end='\r')

            # Initialize array of distances that will be updated with each separate feature group of the case
            distances_i = np.zeros(shape=(self.n_entries))

            # Initializes set of feature prefixes to iterate distances from
            feature_map_i = self.feature_map.copy()

            # Deals with the distances of every separte feature group
            for prefix, opts in feature_map_i.items():
                # Get feature opts
                metric_i, norm_i, weight_i = opts

                # Select appropriate columns for the feature group in df (assumes binarized already)
                f_cnames, f_arr_cb = self.cb_arr_features[prefix]

                # Selects data from enrty i and sorts out the columns belonging to the feature group
                # NOTE: Ranged select to force getting a dataframe instead of a series when iterating individual rows
                f_arr_x = self._feature_prepro(df_x.iloc[i:i+1], f_cnames)

                # Compute distance
                dist_i = scipy.spatial.distance.cdist(f_arr_x, f_arr_cb, metric=metric_i)
                
                if norm_i:
                    # Distance normalization
                    dist_i = dist_i / np.linalg.norm(dist_i, axis=1)[:, np.newaxis]
                
                # Distance weighting
                dist_i *= weight_i

                # Update global distance for query
                distances_i += dist_i[0]

            # Once all feature distances have been resolved, update k *nearest* distances and idx for i
            nearest_k_idx_i = np.argsort(distances_i)[:k]
            nearest_idx.append(nearest_k_idx_i)
            nearest_distances.append(distances_i[nearest_k_idx_i])
            #nearest_distances.append(distances_i[nearest_idx])  #BUG!

        if verbose: print('\n')
        return nearest_distances, nearest_idx

    def aggregate(self, ndists, nidxs, weighted_voting=True, strat='count'):
        # NOTE TODO: This should be carried out from the CBR model. Implement method there
        y_candidate_sols = []
        for i_ndists, i_nidxs in zip(ndists, nidxs):

            # TODO: Confirm this work ok when k=1 (no df to series autocast)
            i_cb_sols = self.cb_y[i_nidxs]

            # Expecting shame number of dimensions
            if len(i_ndists.shape) == 1:
                i_ndists = i_ndists[:, np.newaxis]

            if strat == 'avg':
                # Scaling values from {0, 1} to {-1, 1 for averaging voting}
                i_cb_sols *= 2
                i_cb_sols -= 1    

            if weighted_voting:
                # NOTE: (!) Distances must be normalized for this to work properly
                i_cb_sols = np.multiply(i_cb_sols, (1-i_ndists))
                # NOTE TODO: Check nearest_sols = nearest_sols * (1 - q_dist[nearest_cb_idx]) at fit. That part might fail when targets with more than 1 value...

            if strat == 'count':
                # Round to neares integer (only makes a different when weighted voting is true)
                i_cb_sols = np.round(i_cb_sols)

                # Cast to bool
                i_cb_sols = i_cb_sols.astype(dtype=bool)

                # Count values
                unique, counts = np.unique(i_cb_sols, return_counts=True)
                
                # Keep the most popular solution among neighbours
                i_cb_sols = unique[np.argsort(counts)[-1]]

            # TODO: Implement this in fit()
            elif strat == 'avg':
                # Adds values row-wise, then assumes positive label if the result is positive
                i_cb_sols = np.sum(i_cb_sols, axis=0)
                i_cb_sols = i_cb_sols >= 0

            y_candidate_sols.append(i_cb_sols)

        # Return aggregated solutions
        return np.array(y_candidate_sols)

# TODO:
def simple_aggregate(dists, neighs, weighted_voting=True):
    """ Simple aggregator. Use in combination with PRR's find.
    weighting_voting behaves the same as the parameter with the same name in the find method """
    pass