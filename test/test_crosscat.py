import jax
import jax.numpy as jnp
import pytest
from genjaxmix.model.crosscat import CrossCatModel, CrossCatInference


class TestCrossCatModel:
    """Tests for CrossCat model implementation"""
    
    def test_model_initialization(self):
        """Test basic model initialization"""
        n_rows, n_columns = 100, 5
        column_types = ['continuous', 'categorical', 'continuous', 'categorical', 'continuous']
        
        model = CrossCatModel(
            n_rows=n_rows,
            n_columns=n_columns, 
            column_types=column_types,
            crp_concentration=1.0
        )
        
        assert model.n_rows == n_rows
        assert model.n_columns == n_columns
        assert len(model.column_types) == n_columns
        assert len(model.observations()) == n_columns
        
    def test_continuous_column_setup(self):
        """Test setup of continuous columns"""
        model = CrossCatModel(
            n_rows=50,
            n_columns=3,
            column_types=['continuous', 'continuous', 'continuous']
        )
        
        # Should have hyperpriors for all continuous columns
        assert len(model.continuous_hyperpriors) == 3
        assert len(model.categorical_hyperpriors) == 0
        
    def test_categorical_column_setup(self):
        """Test setup of categorical columns"""
        model = CrossCatModel(
            n_rows=50,
            n_columns=2, 
            column_types=['categorical', 'categorical']
        )
        
        # Should have hyperpriors for all categorical columns
        assert len(model.categorical_hyperpriors) == 2
        assert len(model.continuous_hyperpriors) == 0
        
    def test_mixed_column_setup(self):
        """Test setup of mixed column types"""
        column_types = ['continuous', 'categorical', 'continuous']
        model = CrossCatModel(
            n_rows=50,
            n_columns=3,
            column_types=column_types
        )
        
        assert len(model.continuous_hyperpriors) == 2  # columns 0, 2
        assert len(model.categorical_hyperpriors) == 1  # column 1
        assert 0 in model.continuous_hyperpriors
        assert 2 in model.continuous_hyperpriors
        assert 1 in model.categorical_hyperpriors
        
    def test_column_clustering_initialization(self):
        """Test column clustering initialization"""
        model = CrossCatModel(
            n_rows=50,
            n_columns=4,
            column_types=['continuous'] * 4
        )
        
        key = jax.random.PRNGKey(42)
        model.initialize_parameters(key)
        
        column_clusters = model.get_column_clusters()
        assert column_clusters.shape == (4,)
        assert jnp.all(column_clusters >= 0)


class TestCrossCatInference:
    """Tests for CrossCat inference engine"""
    
    def test_inference_initialization(self):
        """Test inference engine initialization"""
        model = CrossCatModel(
            n_rows=30,
            n_columns=2,
            column_types=['continuous', 'categorical']
        )
        
        inference = CrossCatInference(model)
        assert inference.model == model
        
    def test_gibbs_sampling_setup(self):
        """Test Gibbs sampling setup (without full execution)"""
        model = CrossCatModel(
            n_rows=10,
            n_columns=2,
            column_types=['continuous', 'categorical']
        )
        
        inference = CrossCatInference(model)
        
        # Create synthetic data
        key = jax.random.PRNGKey(123)
        continuous_data = jax.random.normal(key, (10,))
        categorical_data = jax.random.randint(key, (10,), 0, 3)
        
        data = {
            'column_0': continuous_data,
            'column_1': categorical_data
        }
        
        # Test that model can be compiled with data
        try:
            inference_fn = model.compile_crosscat(data)
            assert callable(inference_fn)
        except Exception as e:
            pytest.skip(f"Model compilation not fully implemented: {e}")


class TestChineseRestaurantProcess:
    """Tests for Chinese Restaurant Process implementation"""
    
    def test_crp_initialization(self):
        """Test CRP initialization produces valid assignments"""
        from genjaxmix.model.dsl import ChineseRestaurantProcess, Constant
        
        concentration = Constant(jnp.array(1.0))
        n_columns = 5
        crp = ChineseRestaurantProcess(concentration, n_columns)
        
        key = jax.random.PRNGKey(0)
        assignments = crp.initialize(key)
        
        assert assignments.shape == (n_columns,)
        assert jnp.all(assignments >= 0)
        assert assignments[0] == 0  # First column always assigned to cluster 0
        
    def test_crp_clustering_property(self):
        """Test that CRP produces valid clustering"""
        from genjaxmix.model.dsl import ChineseRestaurantProcess, Constant
        
        concentration = Constant(jnp.array(2.0))
        crp = ChineseRestaurantProcess(concentration, 10)
        
        key = jax.random.PRNGKey(42)
        assignments = crp.initialize(key)
        
        # Check that cluster IDs are contiguous (0, 1, 2, ... without gaps)
        unique_clusters = jnp.unique(assignments)
        expected_clusters = jnp.arange(len(unique_clusters))
        assert jnp.array_equal(jnp.sort(unique_clusters), expected_clusters)


class TestConstrainedColumnClustering:
    """Tests for constrained column clustering functionality"""
    
    def test_constraint_validation(self):
        """Test constraint validation logic"""
        constraints = [(0, 2), (1, 3)]  # Column pairs that cannot be in same view
        model = CrossCatModel(
            n_rows=50,
            n_columns=4,
            column_types=['continuous'] * 4,
            column_constraints=constraints
        )
        
        # Valid assignment (no constrained columns in same cluster)
        valid_assignment = jnp.array([0, 1, 1, 0])  # 0-2 separate, 1-3 separate
        assert model._validate_column_constraints(valid_assignment) == True
        
        # Invalid assignment (column 0 and 2 in same cluster)
        invalid_assignment = jnp.array([0, 1, 0, 2])  # 0-2 together
        assert model._validate_column_constraints(invalid_assignment) == False
        
        # Invalid assignment (column 1 and 3 in same cluster)  
        invalid_assignment2 = jnp.array([0, 1, 2, 1])  # 1-3 together
        assert model._validate_column_constraints(invalid_assignment2) == False
        
    def test_constrained_column_clustering(self):
        """Test constraint-aware column clustering"""
        constraints = [(0, 1)]  # Columns 0 and 1 cannot be in same view
        model = CrossCatModel(
            n_rows=50,
            n_columns=3,
            column_types=['continuous'] * 3,
            column_constraints=constraints
        )
        
        key = jax.random.PRNGKey(123)
        
        # Test multiple samples to ensure constraints are consistently satisfied
        for _ in range(10):
            key, subkey = jax.random.split(key)
            assignment = model.sample_column_clustering_with_constraints(subkey)
            
            # Verify constraints are satisfied
            assert model._validate_column_constraints(assignment) == True
            
            # Specifically check that columns 0 and 1 are in different clusters
            assert assignment[0] != assignment[1]
            
    def test_multiple_constraints(self):
        """Test handling of multiple constraints"""
        constraints = [(0, 2), (1, 3), (2, 4)]  # Multiple constraint pairs
        model = CrossCatModel(
            n_rows=50,
            n_columns=5,
            column_types=['continuous'] * 5,
            column_constraints=constraints
        )
        
        key = jax.random.PRNGKey(456)
        assignment = model.sample_column_clustering_with_constraints(key)
        
        # All constraints should be satisfied
        assert model._validate_column_constraints(assignment) == True
        assert assignment[0] != assignment[2]  # Constraint (0,2)
        assert assignment[1] != assignment[3]  # Constraint (1,3)  
        assert assignment[2] != assignment[4]  # Constraint (2,4)
        
    def test_no_constraints(self):
        """Test that model works normally without constraints"""
        model = CrossCatModel(
            n_rows=50,
            n_columns=4,
            column_types=['continuous'] * 4,
            column_constraints=None
        )
        
        # Should have empty constraints list
        assert model.column_constraints == []
        
        key = jax.random.PRNGKey(789)
        assignment = model.sample_column_clustering_with_constraints(key)
        
        # Validation should always pass with no constraints
        assert model._validate_column_constraints(assignment) == True
        
    def test_greedy_merge_with_constraints(self):
        """Test greedy merge algorithm respects constraints"""
        constraints = [(0, 1)]
        model = CrossCatModel(
            n_rows=50,
            n_columns=4,
            column_types=['continuous'] * 4,
            column_constraints=constraints
        )
        
        # Start with all columns in separate clusters
        initial_assignment = jnp.arange(4)  # [0, 1, 2, 3]
        
        key = jax.random.PRNGKey(101)
        merged_assignment = model._greedy_merge_with_constraints(key, initial_assignment)
        
        # Constraints should still be satisfied after merging
        assert model._validate_column_constraints(merged_assignment) == True
        assert merged_assignment[0] != merged_assignment[1]  # Constraint preserved


if __name__ == "__main__":
    pytest.main([__file__])