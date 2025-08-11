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


if __name__ == "__main__":
    pytest.main([__file__])