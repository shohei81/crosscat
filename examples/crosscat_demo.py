"""
CrossCat Demo: Tabular data analysis with mixed data types

This example demonstrates how to use the CrossCat implementation
for analyzing tabular data with both continuous and categorical columns.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from genjaxmix.model.crosscat import CrossCatModel, CrossCatInference


def generate_synthetic_data(
    key: jax.random.PRNGKey, n_rows: int = 100, constrained_generation: bool = False
) -> dict:
    """Generate synthetic mixed-type tabular data for demonstration"""

    key1, key2, key3, key4, key5 = jax.random.split(key, 5)

    if constrained_generation:
        # Generate data where column 0 and 2 are forced to be independent
        # (simulating constraint that height and weight can't be in same view)

        # Column 0: Height (continuous) - independent clusters
        height = jax.random.normal(key1, (n_rows,)) * 10 + 170

        # Column 1: Gender (categorical) - correlated with height
        height_norm = (height - jnp.mean(height)) / jnp.std(height)
        gender_prob = 1 / (1 + jnp.exp(-height_norm))  # Sigmoid
        gender = jax.random.bernoulli(key2, gender_prob, (n_rows,)).astype(jnp.int32)

        # Column 2: Weight (continuous) - INDEPENDENT from height due to constraint
        weight = jax.random.normal(key3, (n_rows,)) * 15 + 70

        # Column 3: Favorite Color (categorical) - independent
        color = jax.random.categorical(key4, jnp.log(jnp.ones(4) / 4), shape=(n_rows,))
    else:
        # Original correlated data generation
        # Column 0: Height (continuous) - two clusters (short/tall people)
        height_cluster = jax.random.bernoulli(key1, 0.6, (n_rows,))
        height = jnp.where(
            height_cluster,
            jax.random.normal(key1, (n_rows,)) * 5 + 170,  # Tall cluster
            jax.random.normal(key2, (n_rows,)) * 4 + 160,  # Short cluster
        )

        # Column 1: Gender (categorical) - correlated with height
        gender_tall = jax.random.categorical(
            key3, jnp.log(jnp.array([0.3, 0.7])), shape=(n_rows,)
        )  # M/F for tall
        gender_short = jax.random.categorical(
            key4, jnp.log(jnp.array([0.7, 0.3])), shape=(n_rows,)
        )  # M/F for short
        gender = jnp.where(height > 165, gender_tall, gender_short)

        # Column 2: Weight (continuous) - correlated with height
        weight = height * 0.8 + jax.random.normal(key2, (n_rows,)) * 5 - 50

        # Column 3: Favorite Color (categorical) - independent
        color = jax.random.categorical(key4, jnp.log(jnp.ones(4) / 4), shape=(n_rows,))

    return {
        "column_0": height,  # Height (continuous)
        "column_1": gender,  # Gender (categorical)
        "column_2": weight,  # Weight (continuous)
        "column_3": color,  # Color (categorical)
    }


def run_crosscat_analysis(use_constraints: bool = False):
    """Run CrossCat analysis on synthetic data"""

    print("🔬 CrossCat Demo: Mixed Tabular Data Analysis")
    print("=" * 50)

    # Setup
    key = jax.random.PRNGKey(42)
    n_rows = 50  # Reduced for faster execution
    n_columns = 4
    column_types = ["continuous", "categorical", "continuous", "categorical"]

    # Generate data
    print("📊 Generating synthetic tabular data...")
    data_key, model_key = jax.random.split(key)
    data = generate_synthetic_data(
        data_key, n_rows, constrained_generation=use_constraints
    )

    print(f"   - Rows: {n_rows}")
    print(f"   - Columns: {n_columns}")
    print(f"   - Types: {column_types}")

    # Define constraints if requested
    column_constraints = (
        [(0, 2)] if use_constraints else None
    )  # Height and Weight cannot be in same view

    # Create CrossCat model
    print("\n🏗️  Building CrossCat model...")
    if use_constraints:
        print(f"   - Column constraints: {column_constraints}")
    model = CrossCatModel(
        n_rows=n_rows,
        n_columns=n_columns,
        column_types=column_types,
        crp_concentration=1.0,
        column_constraints=column_constraints,
    )

    # Model is now ready (initialization happens in inference)

    print("   ✅ Model initialized successfully")
    print(f"   - Continuous columns: {len(model.continuous_hyperpriors)}")
    print(f"   - Categorical columns: {len(model.categorical_hyperpriors)}")

    # Test column clustering initialization
    column_clusters = model.get_column_clusters()
    n_column_clusters = len(jnp.unique(column_clusters))

    print(f"\n🔗 Initial column clustering:")
    print(f"   - Column assignments: {column_clusters}")
    print(f"   - Number of column clusters: {n_column_clusters}")

    # Test constraint-aware column clustering if constraints are specified
    if use_constraints:
        print(f"\n🚫 Testing constraint-aware column clustering:")
        constraint_key, _ = jax.random.split(model_key)
        constrained_clusters = model.sample_column_clustering_with_constraints(
            constraint_key
        )
        constraint_valid = model._validate_column_constraints(constrained_clusters)
        print(f"   - Constrained assignments: {constrained_clusters}")
        print(f"   - Constraints satisfied: {constraint_valid}")
        print(
            f"   - Column 0 and 2 in same view: {constrained_clusters[0] == constrained_clusters[2]}"
        )

    # Test hyperparameter estimation
    print(f"\n🧮 Hyperparameter estimation:")
    hyperparams = model.update_hyperparameters(model_key, data)
    for col_idx, params in hyperparams.items():
        col_type = column_types[col_idx]
        print(f"   - column_{col_idx} ({col_type}): {list(params.keys())}")

    # Setup and run inference
    print("\n⚡ Setting up inference engine...")
    inference = CrossCatInference(model)

    try:
        print("   - Running Gibbs sampling...")
        
        # Run actual Gibbs sampling
        inference_key, _ = jax.random.split(model_key)
        samples = inference.gibbs_sample(
            inference_key, 
            data, 
            n_iterations=20,
            burn_in=3,
            thin=3
        )
        
        print("   ✅ Gibbs sampling completed successfully!")
        print(f"   - Collected {len(samples['column_clusters'])} samples")
        
        # Show final clustering results
        if len(samples['column_clusters']) > 0:
            final_column_clustering = samples['column_clusters'][-1]
            n_views = len(jnp.unique(final_column_clustering))
            print(f"   - Final column clustering: {final_column_clustering}")
            print(f"   - Number of discovered views: {n_views}")
            
            # Show row clustering for each view
            for view_id in jnp.unique(final_column_clustering):
                if int(view_id) in samples['row_clusters']:
                    row_clusters = samples['row_clusters'][int(view_id)][-1]
                    n_row_clusters = len(jnp.unique(row_clusters))
                    cols_in_view = jnp.where(final_column_clustering == view_id)[0]
                    print(f"   - View {view_id}: columns {list(cols_in_view)}, {n_row_clusters} row clusters")

        # Demonstrate data access
        print(f"\n📈 Data summary:")
        for col_name, col_data in data.items():
            col_idx = int(col_name.split("_")[1])
            col_type = column_types[col_idx]

            if col_type == "continuous":
                print(
                    f"   - {col_name} ({col_type}): mean={jnp.mean(col_data):.2f}, std={jnp.std(col_data):.2f}"
                )
            else:
                unique_vals = len(jnp.unique(col_data))
                print(f"   - {col_name} ({col_type}): {unique_vals} unique values")

    except Exception as e:
        print(f"   ⚠️  Error during inference: {e}")
        import traceback
        traceback.print_exc()

    print("\n✨ CrossCat demo completed!")
    return model, data


def visualize_data(data: dict, save_path: str = None):
    """Visualize the synthetic data"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("CrossCat Demo: Synthetic Tabular Data", fontsize=16)

    # Height distribution
    axes[0, 0].hist(data["column_0"], bins=30, alpha=0.7, color="skyblue")
    axes[0, 0].set_title("Height (Continuous)")
    axes[0, 0].set_xlabel("Height")
    axes[0, 0].set_ylabel("Frequency")

    # Gender distribution
    gender_counts = jnp.bincount(data["column_1"])
    axes[0, 1].bar(["Female", "Male"], gender_counts, color=["pink", "lightblue"])
    axes[0, 1].set_title("Gender (Categorical)")
    axes[0, 1].set_ylabel("Count")

    # Weight distribution
    axes[1, 0].hist(data["column_2"], bins=30, alpha=0.7, color="lightgreen")
    axes[1, 0].set_title("Weight (Continuous)")
    axes[1, 0].set_xlabel("Weight")
    axes[1, 0].set_ylabel("Frequency")

    # Color distribution
    color_counts = jnp.bincount(data["column_3"])
    colors = ["red", "blue", "green", "orange"]
    axes[1, 1].bar(
        colors[: len(color_counts)], color_counts, color=colors[: len(color_counts)]
    )
    axes[1, 1].set_title("Favorite Color (Categorical)")
    axes[1, 1].set_ylabel("Count")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"📊 Visualization saved to {save_path}")
    else:
        plt.show()


def run_constraint_comparison_demo():
    """Run comparison between constrained and unconstrained models"""
    print("🆚 Constraint Comparison Demo")
    print("=" * 60)

    print("\n1️⃣  Running UNCONSTRAINED CrossCat analysis...")
    model_unconstrained, data_unconstrained = run_crosscat_analysis(
        use_constraints=False
    )

    print("\n" + "=" * 60)
    print("\n2️⃣  Running CONSTRAINED CrossCat analysis...")
    model_constrained, data_unconstrained = run_crosscat_analysis(use_constraints=True)

    print("\n" + "=" * 60)
    print("\n📊 Summary Comparison:")
    print("   - Unconstrained model: Normal height-weight correlation")
    print("   - Constrained model: Height and weight forced independent")
    print("   - Constraint ensures columns 0 and 2 never appear in same view")

    return (model_unconstrained, data_unconstrained), (
        model_constrained,
        data_unconstrained,  # Fix: use same data for both models
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # Run comparison demo
        results = run_constraint_comparison_demo()
    else:
        # Run single demo (default: no constraints)
        use_constraints = len(sys.argv) > 1 and sys.argv[1] == "--constrained"
        model, data = run_crosscat_analysis(use_constraints=use_constraints)

        # Optional: Create visualization if matplotlib is available
        try:
            visualize_data(data)
        except ImportError:
            print("📊 Install matplotlib for data visualization")
