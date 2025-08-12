#!/usr/bin/env python3
"""
Simple test of CrossCat implementation
"""

import jax
import jax.numpy as jnp
from genjaxmix.model.crosscat import CrossCatModel, CrossCatInference

def test_simple_crosscat():
    """Test basic CrossCat functionality with minimal data"""
    print("🧪 Simple CrossCat Test")
    print("=" * 30)
    
    # Very small test case
    key = jax.random.PRNGKey(42)
    n_rows, n_columns = 20, 2
    column_types = ['continuous', 'categorical']
    
    # Generate simple test data
    data_key, model_key = jax.random.split(key)
    data = {
        'column_0': jax.random.normal(data_key, (n_rows,)),
        'column_1': jax.random.randint(jax.random.split(data_key)[0], (n_rows,), 0, 2)
    }
    
    print(f"📊 Data: {n_rows} rows, {n_columns} columns")
    print(f"   - Column 0 (continuous): mean={jnp.mean(data['column_0']):.2f}")
    print(f"   - Column 1 (categorical): {len(jnp.unique(data['column_1']))} categories")
    
    # Create model
    model = CrossCatModel(
        n_rows=n_rows,
        n_columns=n_columns,
        column_types=column_types,
        crp_concentration=1.0
    )
    
    print("✅ Model created")
    
    # Test initialization
    model.initialize_from_data(model_key, data)
    print(f"✅ Initialized - Column assignments: {model.column_assignments}")
    print(f"   Row assignments: {list(model.row_assignments.keys())}")
    
    # Test single Gibbs update
    inference = CrossCatInference(model)
    
    # Test column update
    test_key, _ = jax.random.split(model_key)
    old_assignment = model.column_assignments[0]
    new_assignment = model.gibbs_update_column_assignment(test_key, data, 0)
    print(f"✅ Column 0 assignment: {old_assignment} → {new_assignment}")
    
    # Test row update
    for view_id in model.row_assignments:
        old_rows = model.row_assignments[view_id].copy()
        model.gibbs_update_row_clustering(test_key, data, view_id)
        print(f"✅ View {view_id} row updates: changed from {len(jnp.unique(old_rows))} to {len(jnp.unique(model.row_assignments[view_id]))} clusters")
    
    # Test very short inference
    print("🚀 Testing mini inference...")
    try:
        samples = inference.gibbs_sample(
            test_key, 
            data, 
            n_iterations=5,
            burn_in=1,
            thin=1
        )
        print(f"✅ Mini inference complete! {len(samples['column_clusters'])} samples")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_crosscat()
    if success:
        print("\n🎉 CrossCat implementation working!")
    else:
        print("\n❌ Issues found in implementation")