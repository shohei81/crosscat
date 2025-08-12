import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional
from genjaxmix.model.model import Model
from genjaxmix.model.dsl import (
    ChineseRestaurantProcess, 
    Constant, 
    NormalInverseGamma,
    DirichletCategorical
)


class CrossCatModel(Model):
    """CrossCat model for tabular data with mixed data types"""
    
    # Class constants
    DEFAULT_MAX_SAMPLING_ATTEMPTS = 1000
    
    def __init__(self, n_rows: int, n_columns: int, 
                 column_types: List[str], 
                 crp_concentration: float = 1.0,
                 column_constraints: Optional[List[Tuple[int, int]]] = None,
                 max_sampling_attempts: Optional[int] = None):
        """
        Initialize CrossCat model
        
        Args:
            n_rows: Number of rows in the data
            n_columns: Number of columns in the data  
            column_types: List of column types ('continuous' or 'categorical')
            crp_concentration: Concentration parameter for CRP
            column_constraints: List of column pairs (i, j) that cannot be in same view
            max_sampling_attempts: Maximum attempts for constraint-aware sampling (default: 1000)
        """
        super().__init__()
        
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.column_types = column_types
        self.column_constraints = column_constraints or []
        self.max_sampling_attempts = max_sampling_attempts or self.DEFAULT_MAX_SAMPLING_ATTEMPTS
        
        # Convert constraints to set of tuples for O(1) lookup
        self._constraint_set = set(self.column_constraints) if self.column_constraints else set()
        
        # Column clustering via Chinese Restaurant Process
        self.column_clustering = ChineseRestaurantProcess(
            concentration=jnp.array([[crp_concentration]]),  # 2D format
            n_columns=n_columns
        )
        
        # Hyperparameters for continuous columns (Normal-InverseGamma)
        self.continuous_hyperpriors = {}
        
        # Hyperparameters for categorical columns (Dirichlet)
        self.categorical_hyperpriors = {}
        
        # Internal state for inference
        self.column_assignments = None
        self.row_assignments = {}  # Dict[view_id, jnp.ndarray]
        self.hyperparams = {}
        
        # Create column-specific models
        self._setup_column_models()
        
    def _setup_column_models(self):
        """Setup priors for each column based on its type"""
        for i, col_type in enumerate(self.column_types):
            if col_type == 'continuous':
                # Normal-InverseGamma hyperpriors for continuous data
                self.continuous_hyperpriors[i] = NormalInverseGamma(
                    alpha=Constant(jnp.array([[1.0]])),  # Shape parameter
                    beta=Constant(jnp.array([[1.0]])),   # Rate parameter  
                    mu=Constant(jnp.array([[0.0]])),     # Mean prior mean
                    sigma=Constant(jnp.array([[1.0]]))   # Mean prior variance
                )
            elif col_type == 'categorical':
                # Dirichlet hyperpriors for categorical data (assume 10 categories max)
                self.categorical_hyperpriors[i] = DirichletCategorical(
                    alpha=Constant(jnp.ones((1, 10)))  # Symmetric Dirichlet
                )
    
    def observations(self):
        """Define which variables are observed (data columns)"""
        return [f'column_{i}' for i in range(self.n_columns)]
    
    def initialize_from_data(self, key: jax.random.PRNGKey, data: Dict[str, jnp.ndarray]):
        """
        Initialize CrossCat model state from data
        
        Args:
            key: Random key
            data: Dictionary mapping column names to data arrays
        """
        # Initialize column clustering
        if self.column_constraints:
            self.column_assignments = self.sample_column_clustering_with_constraints(key)
        else:
            # Simple initialization: all columns in one view initially
            self.column_assignments = jnp.zeros(self.n_columns, dtype=jnp.int32)
        
        # Initialize row clustering for each view
        unique_views = jnp.unique(self.column_assignments)
        for view_id in unique_views:
            # Start with all rows in one cluster per view
            self.row_assignments[int(view_id)] = jnp.zeros(self.n_rows, dtype=jnp.int32)
        
        # Initialize hyperparameters
        self.hyperparams = self.update_hyperparameters(key, data)
    
    def predict(self, key: jax.random.PRNGKey, 
               data: Dict[str, jnp.ndarray],
               missing_indices: Dict[str, List[int]]) -> Dict[str, jnp.ndarray]:
        """
        Predict missing values using CrossCat model
        
        Args:
            key: Random key
            data: Observed data (with missing values)
            missing_indices: Dictionary mapping column names to lists of missing row indices
            
        Returns:
            Dictionary of predictions for missing values
        """
        if self.column_assignments is None:
            return {}
            
        predictions = {}
        
        for col_name, missing_rows in missing_indices.items():
            if col_name not in data or len(missing_rows) == 0:
                continue
                
            col_idx = int(col_name.split('_')[1])
            col_type = self.column_types[col_idx]
            view_id = int(self.column_assignments[col_idx])
            
            # Get row clustering for this view
            if view_id not in self.row_assignments:
                continue
                
            row_clusters = self.row_assignments[view_id]
            col_data = data[col_name]
            
            col_predictions = []
            
            for row_idx in missing_rows:
                key, pred_key = jax.random.split(key)
                
                # Determine which row cluster this missing value should belong to
                # For simplicity, assign to the most likely cluster based on other columns
                row_cluster = self._predict_row_cluster(
                    pred_key, row_idx, view_id, data
                )
                
                # Predict value based on the cluster
                if col_type == 'continuous':
                    pred_value = self._predict_continuous(
                        pred_key, col_data, row_clusters, row_cluster, col_idx
                    )
                else:  # categorical
                    pred_value = self._predict_categorical(
                        pred_key, col_data, row_clusters, row_cluster, col_idx
                    )
                
                col_predictions.append(pred_value)
            
            predictions[col_name] = jnp.array(col_predictions)
        
        return predictions
        
    def _predict_row_cluster(self, key: jax.random.PRNGKey, 
                           row_idx: int, view_id: int,
                           data: Dict[str, jnp.ndarray]) -> int:
        """Predict which row cluster a missing value belongs to"""
        if view_id not in self.row_assignments:
            return 0
            
        row_clusters = self.row_assignments[view_id]
        unique_clusters = jnp.unique(row_clusters)
        
        # Simple heuristic: assign to most common cluster
        cluster_counts = [(jnp.sum(row_clusters == c), c) for c in unique_clusters]
        cluster_counts.sort(reverse=True)
        return int(cluster_counts[0][1])
        
    def _predict_continuous(self, key: jax.random.PRNGKey,
                          col_data: jnp.ndarray, row_clusters: jnp.ndarray,
                          target_cluster: int, col_idx: int) -> float:
        """Predict continuous value for a cluster"""
        cluster_mask = row_clusters == target_cluster
        cluster_data = col_data[cluster_mask]
        cluster_data = cluster_data[~jnp.isnan(cluster_data)]  # Remove NaN values
        
        if len(cluster_data) > 0:
            # Use cluster mean as prediction
            return float(jnp.mean(cluster_data))
        else:
            # Use hyperparameter mean as fallback
            if col_idx in self.hyperparams:
                return float(self.hyperparams[col_idx].get('mu', 0.0))
            return 0.0
            
    def _predict_categorical(self, key: jax.random.PRNGKey,
                           col_data: jnp.ndarray, row_clusters: jnp.ndarray,
                           target_cluster: int, col_idx: int) -> int:
        """Predict categorical value for a cluster"""
        cluster_mask = row_clusters == target_cluster
        cluster_data = col_data[cluster_mask]
        cluster_data = cluster_data[~jnp.isnan(cluster_data)]  # Remove NaN values
        
        if len(cluster_data) > 0:
            # Use most common category in cluster
            unique_vals, counts = jnp.unique(cluster_data, return_counts=True)
            most_common_idx = jnp.argmax(counts)
            return int(unique_vals[most_common_idx])
        else:
            # Random category as fallback
            if col_idx in self.hyperparams:
                n_cats = self.hyperparams[col_idx].get('n_categories', 2)
                return int(jax.random.randint(key, (), 0, n_cats))
            return 0
    
    def get_column_clusters(self) -> jnp.ndarray:
        """Get current column cluster assignments"""
        if hasattr(self, 'environment') and self.environment:
            column_clustering_id = self.node_to_id[self.column_clustering]
            return self.environment[column_clustering_id]
        return jnp.zeros(self.n_columns, dtype=jnp.int32)
    
    def get_row_clusters(self, view_id: int) -> jnp.ndarray:
        """Get row cluster assignments for a specific view"""
        if view_id in self.row_assignments:
            return self.row_assignments[view_id]
        return jnp.zeros(self.n_rows, dtype=jnp.int32)
        
    def set_row_clusters(self, view_id: int, assignments: jnp.ndarray):
        """Set row cluster assignments for a specific view"""
        self.row_assignments[view_id] = assignments
        
    def get_columns_in_view(self, view_id: int) -> List[int]:
        """Get list of column indices in a specific view"""
        if self.column_assignments is None:
            return []
        return jnp.where(self.column_assignments == view_id)[0].tolist()
    
    def update_hyperparameters(self, key: jax.random.PRNGKey, 
                             data: Dict[str, jnp.ndarray]) -> Dict:
        """
        Update hyperparameters using empirical Bayes or full Bayesian approach
        
        Args:
            key: Random key
            data: Observed data
            
        Returns:
            Updated hyperparameters
        """
        updated_hyperparams = {}
        
        for col_idx, col_type in enumerate(self.column_types):
            col_name = f'column_{col_idx}'
            if col_name not in data:
                continue
                
            col_data = data[col_name]
            
            if col_type == 'continuous':
                # Update Normal-InverseGamma hyperparameters based on data
                sample_mean = jnp.mean(col_data)
                sample_var = jnp.var(col_data)
                n_samples = len(col_data)
                
                # Simple empirical Bayes updates
                # Prior mean updated towards sample mean
                posterior_mu = (sample_mean * n_samples + 0.0) / (n_samples + 1)
                
                # Prior variance incorporates sample variance
                posterior_sigma = jnp.sqrt(sample_var / n_samples + 1.0)
                
                # Inverse-Gamma parameters (shape and rate)
                posterior_alpha = 1.0 + n_samples / 2
                posterior_beta = 1.0 + n_samples * sample_var / 2
                
                updated_hyperparams[col_idx] = {
                    'mu': posterior_mu,
                    'sigma': posterior_sigma,
                    'alpha': posterior_alpha,
                    'beta': posterior_beta
                }
                
            elif col_type == 'categorical':
                # Update Dirichlet hyperparameters based on data
                unique_vals, counts = jnp.unique(col_data, return_counts=True)
                n_categories = len(unique_vals)
                
                # Add pseudocounts (symmetric Dirichlet prior)
                posterior_alpha = counts + 1.0
                
                updated_hyperparams[col_idx] = {
                    'alpha': posterior_alpha,
                    'n_categories': n_categories
                }
        
        return updated_hyperparams
    
    def _validate_column_constraints(self, column_assignments: jnp.ndarray) -> bool:
        """
        Validate that column assignments satisfy the constraints
        
        Args:
            column_assignments: Array of column cluster assignments
            
        Returns:
            True if all constraints are satisfied, False otherwise
        """
        for col_i, col_j in self._constraint_set:
            if column_assignments[col_i] == column_assignments[col_j]:
                return False
        return True
    
    def _generate_valid_column_assignment(self, key: jax.random.PRNGKey, 
                                        current_assignments: jnp.ndarray,
                                        column_to_reassign: int) -> int:
        """
        Generate a valid cluster assignment for a column that satisfies constraints
        
        Args:
            key: Random key
            current_assignments: Current column cluster assignments
            column_to_reassign: Index of column to reassign
            
        Returns:
            Valid cluster assignment
        """
        # Get existing clusters
        existing_clusters = jnp.unique(current_assignments)
        max_cluster = jnp.max(existing_clusters) if len(existing_clusters) > 0 else -1
        
        # Find constrained columns (columns that cannot be in same view)
        constrained_columns = []
        for col_i, col_j in self._constraint_set:
            if col_i == column_to_reassign:
                constrained_columns.append(col_j)
            elif col_j == column_to_reassign:
                constrained_columns.append(col_i)
        
        # Get forbidden clusters (clusters containing constrained columns)
        forbidden_clusters = set()
        for constrained_col in constrained_columns:
            if constrained_col < len(current_assignments):
                forbidden_clusters.add(current_assignments[constrained_col])
        
        # Find valid clusters (existing clusters not in forbidden set)
        valid_existing_clusters = []
        for cluster in existing_clusters:
            if cluster not in forbidden_clusters:
                valid_existing_clusters.append(cluster)
        
        # Always allow creating a new cluster
        valid_clusters = valid_existing_clusters + [max_cluster + 1]
        
        # Sample uniformly from valid clusters
        n_valid = len(valid_clusters)
        if n_valid == 0:
            # Fallback: create new cluster
            return max_cluster + 1
        
        cluster_idx = jax.random.randint(key, (), 0, n_valid)
        return valid_clusters[cluster_idx]
    
    def sample_column_clustering_with_constraints(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        """
        Sample column clustering that satisfies constraints using rejection sampling
        
        Args:
            key: Random key
            
        Returns:
            Valid column cluster assignments
        """
        max_attempts = self.max_sampling_attempts
        
        for attempt in range(max_attempts):
            key, subkey = jax.random.split(key)
            
            # Start with all columns in separate clusters if we have constraints
            if self.column_constraints:
                assignments = jnp.arange(self.n_columns)
            else:
                # Sample from CRP normally
                assignments = jax.random.randint(subkey, (self.n_columns,), 0, self.n_columns)
            
            # Greedily merge clusters while respecting constraints
            if self.column_constraints:
                assignments = self._greedy_merge_with_constraints(subkey, assignments)
            
            # Validate final assignment
            if self._validate_column_constraints(assignments):
                return assignments
                
        # Fallback: all columns in separate clusters
        return jnp.arange(self.n_columns)
    
    def _greedy_merge_with_constraints(self, key: jax.random.PRNGKey, 
                                     assignments: jnp.ndarray) -> jnp.ndarray:
        """
        Greedily merge column clusters while respecting constraints
        
        Args:
            key: Random key
            assignments: Initial column assignments (all separate)
            
        Returns:
            Merged assignments respecting constraints
        """
        assignments = jnp.array(assignments)
        n_merges = jax.random.randint(key, (), 0, self.n_columns // 2)
        
        for _ in range(n_merges):
            key, subkey = jax.random.split(key)
            
            # Try to merge two random clusters
            unique_clusters = jnp.unique(assignments)
            if len(unique_clusters) <= 1:
                break
                
            # Sample two different clusters to potentially merge
            cluster_idx = jax.random.choice(subkey, len(unique_clusters), (2,), replace=False)
            cluster_a, cluster_b = unique_clusters[cluster_idx]
            
            # Check if merging these clusters would violate constraints
            cols_in_a = jnp.where(assignments == cluster_a)[0]
            cols_in_b = jnp.where(assignments == cluster_b)[0]
            
            can_merge = True
            for col_a in cols_in_a:
                for col_b in cols_in_b:
                    col_a_int, col_b_int = int(col_a), int(col_b)
                    if (col_a_int, col_b_int) in self._constraint_set or (col_b_int, col_a_int) in self._constraint_set:
                        can_merge = False
                        break
                if not can_merge:
                    break
            
            # Merge if allowed
            if can_merge:
                assignments = jnp.where(assignments == cluster_b, cluster_a, assignments)
        
        return assignments
        
    def gibbs_update_column_assignment(self, key: jax.random.PRNGKey, 
                                     data: Dict[str, jnp.ndarray],
                                     column_idx: int) -> int:
        """
        Gibbs update for a single column's view assignment

        Args:
            key: Random key
            data: Observed data
            column_idx: Index of column to update
            
        Returns:
            New view assignment for the column
        """
        if self.column_assignments is None:
            return 0
            
        # Get current assignments without this column
        current_assignments = self.column_assignments.at[column_idx].set(-1)
        
        # Get existing views and consider new view
        existing_views = jnp.unique(current_assignments)
        existing_views = existing_views[existing_views != -1]
        max_view = jnp.max(existing_views) if len(existing_views) > 0 else -1
        
        # Candidate views: existing + new
        candidate_views = jnp.concatenate([existing_views, jnp.array([max_view + 1])])
        
        # Compute log probabilities for each candidate view
        log_probs = []
        
        for view_id in candidate_views:
            # Check constraint satisfaction
            test_assignment = current_assignments.at[column_idx].set(view_id)
            if not self._validate_column_constraints(test_assignment):
                log_probs.append(-jnp.inf)
                continue
            
            # Compute CRP probability
            n_cols_in_view = jnp.sum(current_assignments == view_id)
            if view_id in existing_views:
                # Existing view
                log_prior = jnp.log(n_cols_in_view)
            else:
                # New view - use CRP concentration
                log_prior = jnp.log(1.0)  # Simplified CRP
            
            # Compute data likelihood for this assignment
            log_likelihood = self._compute_column_likelihood(
                column_idx, view_id, data
            )
            
            log_probs.append(log_prior + log_likelihood)
        
        # Sample from posterior
        log_probs = jnp.array(log_probs)
        log_probs = log_probs - jnp.max(log_probs)  # Numerical stability
        probs = jnp.exp(log_probs)
        probs = probs / jnp.sum(probs)
        
        chosen_idx = jax.random.categorical(key, jnp.log(probs))
        return int(candidate_views[chosen_idx])
        
    def _compute_column_likelihood(self, column_idx: int, view_id: int, 
                                 data: Dict[str, jnp.ndarray]) -> float:
        """
        Compute likelihood of column data given view assignment

        Args:
            column_idx: Column index
            view_id: View ID to assign column to  
            data: Observed data
            
        Returns:
            Log likelihood
        """
        col_name = f'column_{column_idx}'
        if col_name not in data:
            return 0.0
            
        col_data = data[col_name]
        col_type = self.column_types[column_idx]
        
        # Get row clusters for this view
        view_id_int = int(view_id)
        if view_id_int in self.row_assignments:
            row_clusters = self.row_assignments[view_id_int]
        else:
            row_clusters = jnp.zeros(self.n_rows, dtype=jnp.int32)
        
        # Compute likelihood for each row cluster
        unique_row_clusters = jnp.unique(row_clusters)
        log_likelihood = 0.0
        
        for row_cluster_id in unique_row_clusters:
            cluster_mask = row_clusters == row_cluster_id
            cluster_data = col_data[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
                
            if col_type == 'continuous':
                # Normal-InverseGamma likelihood
                if len(cluster_data) > 1:
                    sample_mean = jnp.mean(cluster_data)
                    sample_var = jnp.var(cluster_data)
                    n = len(cluster_data)
                    
                    # Marginal likelihood under Normal-InverseGamma prior
                    # Using conjugate prior updates
                    alpha0, beta0 = 1.0, 1.0  # Prior hyperparameters
                    mu0, sigma0 = 0.0, 1.0
                    
                    # Posterior parameters (only those needed for marginal likelihood)
                    alpha_n = alpha0 + n/2
                    beta_n = beta0 + 0.5*n*sample_var + 0.5*sigma0*n*(sample_mean - mu0)**2/(sigma0 + n)
                    
                    # Compute the log marginal likelihood for a cluster under a Normal-InverseGamma prior,
                    # resulting in a Student-t marginal likelihood (see e.g. Murphy, "Machine Learning: A Probabilistic Perspective", Eq. 4.202).
                    log_likelihood += float(-n/2 * jnp.log(2*jnp.pi) + 
                                          0.5*jnp.log(sigma0/(sigma0 + n)) +
                                          alpha0*jnp.log(beta0) - alpha_n*jnp.log(beta_n) +
                                          jax.scipy.special.gammaln(alpha_n) - jax.scipy.special.gammaln(alpha0))
                else:
                    log_likelihood += 0.0  # Single point
                    
            elif col_type == 'categorical':
                # Dirichlet-Categorical likelihood
                unique_vals, counts = jnp.unique(cluster_data, return_counts=True)
                
                # Marginal likelihood under Dirichlet prior
                alpha = 1.0  # Symmetric Dirichlet parameter
                n = len(cluster_data)
                k = len(unique_vals)
                
                # Multinomial-Dirichlet marginal likelihood using counts
                log_likelihood += float(
                    jax.scipy.special.gammaln(k * alpha) - 
                    jax.scipy.special.gammaln(n + k * alpha) +
                    jnp.sum(jax.scipy.special.gammaln(counts + alpha) - 
                           jax.scipy.special.gammaln(alpha))
                )
        
        return log_likelihood
        
    def gibbs_update_row_clustering(self, key: jax.random.PRNGKey, 
                                  data: Dict[str, jnp.ndarray], 
                                  view_id: int):
        """
        Gibbs update for row clustering within a view

        Args:
            key: Random key
            data: Observed data
            view_id: View ID to update row clustering for
        """
        if view_id not in self.row_assignments:
            return
            
        columns_in_view = self.get_columns_in_view(view_id)
        if len(columns_in_view) == 0:
            return
            
        current_assignments = self.row_assignments[view_id]
        n_rows = len(current_assignments)
        
        # Update only a subset of rows for efficiency (every 3rd row)
        rows_to_update = list(range(0, n_rows, 3))  # Update every 3rd row
        
        # Update selected rows' cluster assignments
        for row_idx in rows_to_update:
            key, subkey = jax.random.split(key)
            
            # Remove this row temporarily
            temp_assignments = current_assignments.at[row_idx].set(-1)
            
            # Get candidate clusters
            existing_clusters = jnp.unique(temp_assignments)
            existing_clusters = existing_clusters[existing_clusters != -1]
            max_cluster = jnp.max(existing_clusters) if len(existing_clusters) > 0 else -1
            candidate_clusters = jnp.concatenate([existing_clusters, jnp.array([max_cluster + 1])])
            
            # Compute probabilities for each cluster
            log_probs = []
            
            for cluster_id in candidate_clusters:
                # CRP prior
                n_in_cluster = jnp.sum(temp_assignments == cluster_id)
                if cluster_id in existing_clusters:
                    log_prior = jnp.log(n_in_cluster)
                else:
                    log_prior = jnp.log(1.0)  # New cluster
                
                # Data likelihood
                log_likelihood = self._compute_row_likelihood(
                    row_idx, cluster_id, columns_in_view, data, temp_assignments
                )
                
                log_probs.append(log_prior + log_likelihood)
            
            # Sample new assignment
            log_probs = jnp.array(log_probs)
            log_probs = log_probs - jnp.max(log_probs)
            probs = jnp.exp(log_probs)
            probs = probs / jnp.sum(probs)
            
            chosen_idx = jax.random.categorical(subkey, jnp.log(probs))
            new_cluster = int(candidate_clusters[chosen_idx])
            
            # Update assignment
            self.row_assignments[view_id] = current_assignments.at[row_idx].set(new_cluster)
            current_assignments = self.row_assignments[view_id]
            
    def _compute_row_likelihood(self, row_idx: int, cluster_id: int, 
                              columns_in_view: List[int], 
                              data: Dict[str, jnp.ndarray],
                              row_assignments: jnp.ndarray) -> float:
        """
        Compute likelihood of row data given cluster assignment

        Args:
            row_idx: Row index
            cluster_id: Cluster ID to assign row to
            columns_in_view: Columns in this view
            data: Observed data
            row_assignments: Current row assignments
            
        Returns:
            Log likelihood
        """
        log_likelihood = 0.0
        
        for col_idx in columns_in_view:
            col_name = f'column_{col_idx}'
            if col_name not in data:
                continue
                
            col_data = data[col_name]
            col_type = self.column_types[col_idx]
            
            # Get data for this cluster (including the candidate row)
            cluster_mask = row_assignments == cluster_id
            if cluster_id not in jnp.unique(row_assignments):
                # New cluster - only this row
                cluster_data = jnp.array([col_data[row_idx]])
            else:
                # Existing cluster - add this row's data
                cluster_data = jnp.concatenate([
                    col_data[cluster_mask], 
                    jnp.array([col_data[row_idx]])
                ])
            
            if col_type == 'continuous':
                # Use hyperparameters from hyperparams if available
                if col_idx in self.hyperparams:
                    mu = self.hyperparams[col_idx].get('mu', 0.0)
                    sigma = jnp.sqrt(self.hyperparams[col_idx].get('beta', 1.0) / 
                                   self.hyperparams[col_idx].get('alpha', 1.0))
                else:
                    mu, sigma = 0.0, 1.0
                    
                # Likelihood under normal distribution
                log_likelihood += float(jax.scipy.stats.norm.logpdf(
                    col_data[row_idx], mu, sigma
                ))
                
            elif col_type == 'categorical':
                # Categorical likelihood with Dirichlet prior
                if len(cluster_data) > 1:
                    # Use empirical distribution of cluster
                    unique_vals, counts = jnp.unique(cluster_data[:-1], return_counts=True)
                    alpha = 1.0  # Symmetric prior
                    
                    # Probability of observing this category using counts
                    obs_val = int(col_data[row_idx])
                    # Find count for observed value
                    obs_idx = jnp.where(unique_vals == obs_val)[0]
                    count_obs = counts[obs_idx[0]] if len(obs_idx) > 0 else 0
                    
                    total_count = jnp.sum(counts)
                    prob = (count_obs + alpha) / (total_count + alpha * len(unique_vals))
                    log_likelihood += float(jnp.log(prob))
                else:
                    # Single point in cluster - uniform probability
                    log_likelihood += 0.0
        
        return log_likelihood


class CrossCatInference:
    """Inference engine for CrossCat models"""
    
    def __init__(self, model: CrossCatModel):
        self.model = model
        
    def gibbs_sample(self, key: jax.random.PRNGKey, 
                    data: Dict[str, jnp.ndarray],
                    n_iterations: int = 1000,
                    burn_in: int = 100,
                    thin: int = 10) -> Dict[str, List[jnp.ndarray]]:
        """
        Run Gibbs sampling for CrossCat inference
        
        Args:
            key: Random key
            data: Observed data
            n_iterations: Number of Gibbs sampling iterations
            burn_in: Number of burn-in iterations to discard
            thin: Thinning interval for samples
            
        Returns:
            Samples from posterior distributions
        """
        # Initialize model state
        init_key, key = jax.random.split(key)
        self.model.initialize_from_data(init_key, data)
        
        samples = {
            'column_clusters': [],
            'row_clusters': {},
            'hyperparameters': []
        }
        
        print(f"🚀 Starting Gibbs sampling: {n_iterations} iterations, burn-in: {burn_in}, thin: {thin}")
        
        for iteration in range(n_iterations):
            key, iter_key = jax.random.split(key)
            
            # 1. Update column clustering (view assignments)
            self._gibbs_update_column_clustering(iter_key, data)
            
            # 2. Update row clustering within each view (less frequently for performance)
            if iteration % 3 == 0:  # Update row clusters every 3 iterations
                unique_views = jnp.unique(self.model.column_assignments)
                for view_id in unique_views:
                    view_key, iter_key = jax.random.split(iter_key)
                    self.model.gibbs_update_row_clustering(view_key, data, int(view_id))
            
            # 3. Update hyperparameters
            param_key, iter_key = jax.random.split(iter_key)
            self.model.hyperparams = self.model.update_hyperparameters(param_key, data)
            
            # Store samples (after burn-in, with thinning)
            if iteration >= burn_in and iteration % thin == 0:
                if self.model.column_assignments is not None:
                    samples['column_clusters'].append(jnp.array(self.model.column_assignments))
                
                # Store row clusters for each view
                for view_id in unique_views:
                    view_id_int = int(view_id)
                    if view_id_int not in samples['row_clusters']:
                        samples['row_clusters'][view_id_int] = []
                    samples['row_clusters'][view_id_int].append(
                        self.model.get_row_clusters(view_id_int).copy()
                    )
                
                samples['hyperparameters'].append(self.model.hyperparams.copy())
            
            # Progress update
            if (iteration + 1) % 100 == 0:
                n_views = len(jnp.unique(self.model.column_assignments))
                print(f"   Iteration {iteration + 1}/{n_iterations}, Views: {n_views}")
                
        print("✅ Gibbs sampling completed!")
        return samples
    
    def _gibbs_update_column_clustering(self, key: jax.random.PRNGKey, 
                                      data: Dict[str, jnp.ndarray]):
        """
        Update all column assignments using Gibbs sampling
        
        Args:
            key: Random key
            data: Observed data
        """
        n_columns = self.model.n_columns
        keys = jax.random.split(key, n_columns)
        
        for col_idx in range(n_columns):
            new_view = self.model.gibbs_update_column_assignment(
                keys[col_idx], data, col_idx
            )
            
            # Update assignment
            if self.model.column_assignments is not None:
                old_view = self.model.column_assignments[col_idx]
                self.model.column_assignments = self.model.column_assignments.at[col_idx].set(new_view)
            else:
                old_view = 0
            
            # If column moved to a new view, initialize row clustering for that view
            if new_view not in self.model.row_assignments:
                self.model.row_assignments[int(new_view)] = jnp.zeros(self.model.n_rows, dtype=jnp.int32)
            
            # Clean up empty views
            if old_view != new_view:
                columns_in_old_view = jnp.sum(self.model.column_assignments == old_view)
                if columns_in_old_view == 0 and int(old_view) in self.model.row_assignments:
                    del self.model.row_assignments[int(old_view)]