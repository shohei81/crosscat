import jax
import jax.numpy as jnp
from typing import Dict, List, Union, Tuple
from genjaxmix.model.model import Model
from genjaxmix.model.dsl import (
    ChineseRestaurantProcess, 
    Constant, 
    Normal, 
    NormalInverseGamma,
    DirichletCategorical,
    Categorical
)


class CrossCatModel(Model):
    """CrossCat model for tabular data with mixed data types"""
    
    def __init__(self, n_rows: int, n_columns: int, 
                 column_types: List[str], 
                 crp_concentration: float = 1.0):
        """
        Initialize CrossCat model
        
        Args:
            n_rows: Number of rows in the data
            n_columns: Number of columns in the data  
            column_types: List of column types ('continuous' or 'categorical')
            crp_concentration: Concentration parameter for CRP
        """
        super().__init__()
        
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.column_types = column_types
        
        # Column clustering via Chinese Restaurant Process
        self.column_clustering = ChineseRestaurantProcess(
            concentration=jnp.array([[crp_concentration]]),  # 2D format
            n_columns=n_columns
        )
        
        # Hyperparameters for continuous columns (Normal-InverseGamma)
        self.continuous_hyperpriors = {}
        
        # Hyperparameters for categorical columns (Dirichlet)
        self.categorical_hyperpriors = {}
        
        # Create column-specific models
        self._setup_column_models()
        
        # Discover nodes after setup
        self._discover_nodes()
        
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
    
    def compile_crosscat(self, data: Dict[str, jnp.ndarray]):
        """
        Compile CrossCat inference for the given data
        
        Args:
            data: Dictionary mapping column names to data arrays
        """
        # Add data as observed variables
        for i, col_name in enumerate(self.observations()):
            if col_name in data:
                col_data = data[col_name]
                col_type = self.column_types[i]
                
                if col_type == 'continuous':
                    # Create Normal likelihood for continuous data
                    setattr(self, col_name, Normal(
                        mu=Constant(jnp.zeros((self.n_rows, 1))),
                        sigma=Constant(jnp.ones((self.n_rows, 1)))
                    ))
                elif col_type == 'categorical':
                    # Create Categorical likelihood for categorical data
                    n_categories = int(jnp.max(col_data)) + 1
                    probs = jnp.ones((self.n_rows, n_categories)) / n_categories
                    setattr(self, col_name, Categorical(
                        probs=Constant(probs)
                    ))
        
        return self.compile()
    
    def predict(self, key: jax.random.PRNGKey, 
               missing_indices: Dict[str, List[int]]) -> Dict[str, jnp.ndarray]:
        """
        Predict missing values using CrossCat model
        
        Args:
            key: Random key
            missing_indices: Dictionary mapping column names to lists of missing row indices
            
        Returns:
            Dictionary of predictions for missing values
        """
        # This would implement the CrossCat prediction algorithm
        # For now, return empty dict as placeholder
        return {}
    
    def get_column_clusters(self) -> jnp.ndarray:
        """Get current column cluster assignments"""
        if hasattr(self, 'environment') and self.environment:
            column_clustering_id = self.node_to_id[self.column_clustering]
            return self.environment[column_clustering_id]
        return jnp.zeros(self.n_columns, dtype=jnp.int32)
    
    def get_row_clusters(self, column_cluster: int) -> jnp.ndarray:
        """Get row cluster assignments for a specific column cluster"""
        # This would return row clustering for the given column cluster
        # Placeholder implementation
        return jnp.zeros(self.n_rows, dtype=jnp.int32)
    
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


class CrossCatInference:
    """Inference engine for CrossCat models"""
    
    def __init__(self, model: CrossCatModel):
        self.model = model
        
    def gibbs_sample(self, key: jax.random.PRNGKey, 
                    data: Dict[str, jnp.ndarray],
                    n_iterations: int = 1000) -> Dict[str, List[jnp.ndarray]]:
        """
        Run Gibbs sampling for CrossCat inference
        
        Args:
            key: Random key
            data: Observed data
            n_iterations: Number of Gibbs sampling iterations
            
        Returns:
            Samples from posterior distributions
        """
        # Compile model with data
        inference_fn = self.model.compile_crosscat(data)
        
        # Initialize parameters
        self.model.initalize_parameters(key)
        self.model.observe(data)
        
        samples = {
            'column_clusters': [],
            'row_clusters': [],
            'parameters': []
        }
        
        # Initial state
        pi = jnp.ones(10) / 10  # Initial mixture proportions
        assignments = jnp.zeros(self.model.n_rows, dtype=jnp.int32)  # Initial row assignments
        
        for i in range(n_iterations):
            key, subkey = jax.random.split(key)
            
            # Run one step of inference
            environment, assignments, pi = inference_fn(
                subkey, self.model.environment, pi, assignments
            )
            
            # Store samples
            if i % 10 == 0:  # Thin samples
                samples['column_clusters'].append(self.model.get_column_clusters())
                samples['row_clusters'].append(assignments)
                samples['parameters'].append(environment.copy())
                
        return samples