"""
CrossCat Demo: Tabular data analysis with mixed data types

This example demonstrates how to use the CrossCat implementation 
for analyzing tabular data with both continuous and categorical columns.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from genjaxmix.model.crosscat import CrossCatModel, CrossCatInference


def generate_synthetic_data(key: jax.random.PRNGKey, n_rows: int = 100) -> dict:
    """Generate synthetic mixed-type tabular data for demonstration"""
    
    key1, key2, key3, key4 = jax.random.split(key, 4)
    
    # Create synthetic data with known structure
    # Column 0: Height (continuous) - two clusters (short/tall people)
    height_cluster = jax.random.bernoulli(key1, 0.6, (n_rows,))
    height = jnp.where(
        height_cluster,
        jax.random.normal(key1, (n_rows,)) * 5 + 170,  # Tall cluster
        jax.random.normal(key2, (n_rows,)) * 4 + 160   # Short cluster  
    )
    
    # Column 1: Gender (categorical) - correlated with height
    gender_tall = jax.random.categorical(key3, jnp.log(jnp.array([0.3, 0.7])), shape=(n_rows,))  # M/F for tall
    gender_short = jax.random.categorical(key4, jnp.log(jnp.array([0.7, 0.3])), shape=(n_rows,))  # M/F for short
    gender = jnp.where(height > 165, gender_tall, gender_short)
    
    # Column 2: Weight (continuous) - correlated with height
    weight = height * 0.8 + jax.random.normal(key2, (n_rows,)) * 5 - 50
    
    # Column 3: Favorite Color (categorical) - independent
    color = jax.random.categorical(key4, jnp.log(jnp.ones(4) / 4), shape=(n_rows,))
    
    return {
        'column_0': height,    # Height (continuous)
        'column_1': gender,    # Gender (categorical) 
        'column_2': weight,    # Weight (continuous)
        'column_3': color      # Color (categorical)
    }


def run_crosscat_analysis():
    """Run CrossCat analysis on synthetic data"""
    
    print("🔬 CrossCat Demo: Mixed Tabular Data Analysis")
    print("=" * 50)
    
    # Setup
    key = jax.random.PRNGKey(42)
    n_rows = 200
    n_columns = 4
    column_types = ['continuous', 'categorical', 'continuous', 'categorical']
    
    # Generate data
    print("📊 Generating synthetic tabular data...")
    data_key, model_key = jax.random.split(key)
    data = generate_synthetic_data(data_key, n_rows)
    
    print(f"   - Rows: {n_rows}")
    print(f"   - Columns: {n_columns}")
    print(f"   - Types: {column_types}")
    
    # Create CrossCat model
    print("\n🏗️  Building CrossCat model...")
    model = CrossCatModel(
        n_rows=n_rows,
        n_columns=n_columns,
        column_types=column_types,
        crp_concentration=1.0
    )
    
    # Initialize model parameters
    model.initialize_parameters(model_key)
    
    print("   ✅ Model initialized successfully")
    print(f"   - Continuous columns: {len(model.continuous_hyperpriors)}")
    print(f"   - Categorical columns: {len(model.categorical_hyperpriors)}")
    
    # Test column clustering initialization
    column_clusters = model.get_column_clusters()
    n_column_clusters = len(jnp.unique(column_clusters))
    
    print(f"\n🔗 Initial column clustering:")
    print(f"   - Column assignments: {column_clusters}")
    print(f"   - Number of column clusters: {n_column_clusters}")
    
    # Test hyperparameter estimation
    print(f"\n🧮 Hyperparameter estimation:")
    hyperparams = model.update_hyperparameters(model_key, data)
    for col_idx, params in hyperparams.items():
        col_type = column_types[col_idx]
        print(f"   - column_{col_idx} ({col_type}): {list(params.keys())}")
    
    # Setup inference (placeholder - full inference would be implemented here)
    print("\n⚡ Setting up inference engine...")
    inference = CrossCatInference(model)
    
    try:
        # This would run full Gibbs sampling in a complete implementation
        print("   - Inference engine ready")
        print("   - Note: Full Gibbs sampling would be implemented here")
        
        # Demonstrate data access
        print(f"\n📈 Data summary:")
        for col_name, col_data in data.items():
            col_idx = int(col_name.split('_')[1])
            col_type = column_types[col_idx]
            
            if col_type == 'continuous':
                print(f"   - {col_name} ({col_type}): mean={jnp.mean(col_data):.2f}, std={jnp.std(col_data):.2f}")
            else:
                unique_vals = len(jnp.unique(col_data))
                print(f"   - {col_name} ({col_type}): {unique_vals} unique values")
                
    except Exception as e:
        print(f"   ⚠️  Full inference not yet implemented: {e}")
    
    print("\n✨ CrossCat demo completed!")
    return model, data


def visualize_data(data: dict, save_path: str = None):
    """Visualize the synthetic data"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('CrossCat Demo: Synthetic Tabular Data', fontsize=16)
    
    # Height distribution
    axes[0, 0].hist(data['column_0'], bins=30, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Height (Continuous)')
    axes[0, 0].set_xlabel('Height')
    axes[0, 0].set_ylabel('Frequency')
    
    # Gender distribution  
    gender_counts = jnp.bincount(data['column_1'])
    axes[0, 1].bar(['Female', 'Male'], gender_counts, color=['pink', 'lightblue'])
    axes[0, 1].set_title('Gender (Categorical)')
    axes[0, 1].set_ylabel('Count')
    
    # Weight distribution
    axes[1, 0].hist(data['column_2'], bins=30, alpha=0.7, color='lightgreen')  
    axes[1, 0].set_title('Weight (Continuous)')
    axes[1, 0].set_xlabel('Weight')
    axes[1, 0].set_ylabel('Frequency')
    
    # Color distribution
    color_counts = jnp.bincount(data['column_3'])
    colors = ['red', 'blue', 'green', 'orange']
    axes[1, 1].bar(colors[:len(color_counts)], color_counts, color=colors[:len(color_counts)])
    axes[1, 1].set_title('Favorite Color (Categorical)')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Visualization saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Run the demo
    model, data = run_crosscat_analysis()
    
    # Optional: Create visualization if matplotlib is available
    try:
        visualize_data(data)
    except ImportError:
        print("📊 Install matplotlib for data visualization")