"""
Carbon Emissions Predictor - SDG 13: Climate Action
====================================================

This machine learning model predicts carbon emissions based on various
socioeconomic and industrial factors to support climate action initiatives.

Author: [Your Name]
Date: November 28, 2025
SDG: 13 - Climate Action
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. DATA GENERATION
# ============================================================================

def generate_carbon_emissions_data(n_samples=1000):
    """
    Generate synthetic carbon emissions dataset based on real-world patterns.
    
    Features:
    - GDP per capita (USD)
    - Population (millions)
    - Industrial output (% of GDP)
    - Renewable energy usage (%)
    - Forest coverage (%)
    - Vehicle ownership (per 1000 people)
    - Energy consumption (kWh per capita)
    
    Target:
    - Carbon emissions (metric tons per capita)
    """
    print("Generating synthetic carbon emissions dataset...")
    
    # Generate features
    gdp_per_capita = np.random.uniform(1000, 80000, n_samples)
    population = np.random.uniform(0.5, 1400, n_samples)
    industrial_output = np.random.uniform(10, 60, n_samples)
    renewable_energy = np.random.uniform(5, 80, n_samples)
    forest_coverage = np.random.uniform(5, 70, n_samples)
    vehicle_ownership = np.random.uniform(50, 800, n_samples)
    energy_consumption = np.random.uniform(1000, 15000, n_samples)
    
    # Calculate carbon emissions with realistic relationships
    carbon_emissions = (
        0.0001 * gdp_per_capita +
        0.002 * population +
        0.05 * industrial_output -
        0.03 * renewable_energy -
        0.02 * forest_coverage +
        0.008 * vehicle_ownership +
        0.0005 * energy_consumption +
        np.random.normal(0, 1, n_samples)  # Add noise
    )
    
    # Ensure non-negative emissions
    carbon_emissions = np.maximum(carbon_emissions, 0.1)
    
    # Create DataFrame
    data = pd.DataFrame({
        'GDP_per_capita': gdp_per_capita,
        'Population_millions': population,
        'Industrial_output_pct': industrial_output,
        'Renewable_energy_pct': renewable_energy,
        'Forest_coverage_pct': forest_coverage,
        'Vehicle_ownership_per_1000': vehicle_ownership,
        'Energy_consumption_kWh': energy_consumption,
        'Carbon_emissions_tons_per_capita': carbon_emissions
    })
    
    print(f"✓ Generated {n_samples} samples with 7 features")
    return data


# ============================================================================
# 2. DATA EXPLORATION
# ============================================================================

def explore_data(data):
    """Perform exploratory data analysis."""
    print("\n" + "="*70)
    print("DATA EXPLORATION")
    print("="*70)
    
    print("\nDataset Shape:", data.shape)
    print("\nFirst 5 rows:")
    print(data.head())
    
    print("\nStatistical Summary:")
    print(data.describe())
    
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    print("\nCorrelation with Carbon Emissions:")
    correlations = data.corr()['Carbon_emissions_tons_per_capita'].sort_values(ascending=False)
    print(correlations)
    
    return data


# ============================================================================
# 3. DATA VISUALIZATION
# ============================================================================

def visualize_data(data):
    """Create visualizations of the dataset."""
    print("\nGenerating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Correlation heatmap
    plt.subplot(3, 3, 1)
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Heatmap', fontsize=12, fontweight='bold')
    
    # 2. Carbon emissions distribution
    plt.subplot(3, 3, 2)
    plt.hist(data['Carbon_emissions_tons_per_capita'], bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
    plt.xlabel('Carbon Emissions (tons per capita)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Carbon Emissions', fontweight='bold')
    
    # 3. GDP vs Emissions
    plt.subplot(3, 3, 3)
    plt.scatter(data['GDP_per_capita'], data['Carbon_emissions_tons_per_capita'], 
                alpha=0.5, c='#3498db')
    plt.xlabel('GDP per Capita (USD)')
    plt.ylabel('Carbon Emissions')
    plt.title('GDP vs Carbon Emissions', fontweight='bold')
    
    # 4. Renewable Energy vs Emissions
    plt.subplot(3, 3, 4)
    plt.scatter(data['Renewable_energy_pct'], data['Carbon_emissions_tons_per_capita'], 
                alpha=0.5, c='#2ecc71')
    plt.xlabel('Renewable Energy (%)')
    plt.ylabel('Carbon Emissions')
    plt.title('Renewable Energy vs Emissions', fontweight='bold')
    
    # 5. Forest Coverage vs Emissions
    plt.subplot(3, 3, 5)
    plt.scatter(data['Forest_coverage_pct'], data['Carbon_emissions_tons_per_capita'], 
                alpha=0.5, c='#27ae60')
    plt.xlabel('Forest Coverage (%)')
    plt.ylabel('Carbon Emissions')
    plt.title('Forest Coverage vs Emissions', fontweight='bold')
    
    # 6. Industrial Output vs Emissions
    plt.subplot(3, 3, 6)
    plt.scatter(data['Industrial_output_pct'], data['Carbon_emissions_tons_per_capita'], 
                alpha=0.5, c='#e67e22')
    plt.xlabel('Industrial Output (% of GDP)')
    plt.ylabel('Carbon Emissions')
    plt.title('Industrial Output vs Emissions', fontweight='bold')
    
    # 7. Energy Consumption vs Emissions
    plt.subplot(3, 3, 7)
    plt.scatter(data['Energy_consumption_kWh'], data['Carbon_emissions_tons_per_capita'], 
                alpha=0.5, c='#9b59b6')
    plt.xlabel('Energy Consumption (kWh per capita)')
    plt.ylabel('Carbon Emissions')
    plt.title('Energy Consumption vs Emissions', fontweight='bold')
    
    # 8. Vehicle Ownership vs Emissions
    plt.subplot(3, 3, 8)
    plt.scatter(data['Vehicle_ownership_per_1000'], data['Carbon_emissions_tons_per_capita'], 
                alpha=0.5, c='#34495e')
    plt.xlabel('Vehicle Ownership (per 1000 people)')
    plt.ylabel('Carbon Emissions')
    plt.title('Vehicle Ownership vs Emissions', fontweight='bold')
    
    # 9. Feature importance preview
    plt.subplot(3, 3, 9)
    feature_names = data.columns[:-1]
    sample_importance = np.abs(data.corr()['Carbon_emissions_tons_per_capita'][:-1])
    plt.barh(feature_names, sample_importance, color='#1abc9c')
    plt.xlabel('Correlation Strength')
    plt.title('Feature Correlation Strength', fontweight='bold')
    plt.tight_layout()
    
    plt.savefig('visualizations/data_exploration.png', dpi=300, bbox_inches='tight')
    print("✓ Saved data exploration visualizations")
    plt.show()


# ============================================================================
# 4. MODEL TRAINING
# ============================================================================

def train_models(X_train, X_test, y_train, y_test):
    """Train Random Forest and Neural Network models."""
    print("\n" + "="*70)
    print("MODEL TRAINING")
    print("="*70)
    
    # Model 1: Random Forest Regressor
    print("\n[1] Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Model 2: Neural Network (MLP)
    print("[2] Training Neural Network (MLP)...")
    nn_model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    nn_model.fit(X_train, y_train)
    
    print("✓ Models trained successfully")
    return rf_model, nn_model


# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================

def evaluate_models(models, model_names, X_train, X_test, y_train, y_test):
    """Evaluate and compare models."""
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    results = []
    
    for model, name in zip(models, model_names):
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                     scoring='r2', n_jobs=-1)
        
        results.append({
            'Model': name,
            'Train R²': train_r2,
            'Test R²': test_r2,
            'RMSE': test_rmse,
            'MAE': test_mae,
            'CV R² Mean': cv_scores.mean(),
            'CV R² Std': cv_scores.std()
        })
        
        print(f"\n{name}:")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test R²:  {test_r2:.4f}")
        print(f"  RMSE:     {test_rmse:.4f}")
        print(f"  MAE:      {test_mae:.4f}")
        print(f"  CV R²:    {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    results_df = pd.DataFrame(results)
    return results_df


# ============================================================================
# 6. RESULTS VISUALIZATION
# ============================================================================

def visualize_results(models, model_names, X_test, y_test, feature_names):
    """Visualize model predictions and feature importance."""
    print("\nGenerating results visualizations...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Predictions vs Actual for each model
    for idx, (model, name) in enumerate(zip(models, model_names)):
        y_pred = model.predict(X_test)
        
        # Scatter plot
        plt.subplot(2, 3, idx + 1)
        plt.scatter(y_test, y_pred, alpha=0.5, c='#3498db')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Emissions')
        plt.ylabel('Predicted Emissions')
        plt.title(f'{name} - Predictions vs Actual', fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Residuals
        plt.subplot(2, 3, idx + 3)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5, c='#e74c3c')
        plt.axhline(y=0, color='black', linestyle='--', lw=2)
        plt.xlabel('Predicted Emissions')
        plt.ylabel('Residuals')
        plt.title(f'{name} - Residual Plot', fontweight='bold')
        plt.grid(alpha=0.3)
    
    # Feature importance (Random Forest only)
    plt.subplot(2, 3, 5)
    if hasattr(models[0], 'feature_importances_'):
        importances = models[0].feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.barh(range(len(importances)), importances[indices], color='#2ecc71')
        plt.yticks(range(len(importances)), [feature_names[i] for i in indices])
        plt.xlabel('Importance Score')
        plt.title('Random Forest - Feature Importance', fontweight='bold')
        plt.tight_layout()
    
    # Model comparison
    plt.subplot(2, 3, 6)
    model_scores = [r2_score(y_test, model.predict(X_test)) for model in models]
    plt.bar(model_names, model_scores, color=['#3498db', '#e74c3c'])
    plt.ylabel('R² Score')
    plt.title('Model Comparison', fontweight='bold')
    plt.ylim(0, 1)
    for i, score in enumerate(model_scores):
        plt.text(i, score + 0.02, f'{score:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/model_results.png', dpi=300, bbox_inches='tight')
    print("✓ Saved model results visualizations")
    plt.show()


# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("="*70)
    print("CARBON EMISSIONS PREDICTOR - SDG 13: CLIMATE ACTION")
    print("="*70)
    
    # Step 1: Generate data
    data = generate_carbon_emissions_data(n_samples=1000)
    
    # Step 2: Explore data
    data = explore_data(data)
    
    # Step 3: Visualize data
    visualize_data(data)
    
    # Step 4: Prepare data for modeling
    print("\n" + "="*70)
    print("DATA PREPARATION")
    print("="*70)
    
    X = data.drop('Carbon_emissions_tons_per_capita', axis=1)
    y = data['Carbon_emissions_tons_per_capita']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 5: Train models
    rf_model, nn_model = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Step 6: Evaluate models
    models = [rf_model, nn_model]
    model_names = ['Random Forest', 'Neural Network']
    results_df = evaluate_models(models, model_names, X_train_scaled, X_test_scaled, 
                                  y_train, y_test)
    
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Step 7: Visualize results
    visualize_results(models, model_names, X_test_scaled, y_test, X.columns)
    
    # Step 8: Save models
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    import joblib
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    joblib.dump(nn_model, 'models/neural_network_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✓ Models saved to 'models/' directory")
    
    # Step 9: Example prediction
    print("\n" + "="*70)
    print("EXAMPLE PREDICTION")
    print("="*70)
    
    example_country = pd.DataFrame({
        'GDP_per_capita': [45000],
        'Population_millions': [50],
        'Industrial_output_pct': [35],
        'Renewable_energy_pct': [25],
        'Forest_coverage_pct': [30],
        'Vehicle_ownership_per_1000': [600],
        'Energy_consumption_kWh': [8000]
    })
    
    example_scaled = scaler.transform(example_country)
    rf_prediction = rf_model.predict(example_scaled)[0]
    nn_prediction = nn_model.predict(example_scaled)[0]
    
    print("\nExample Country Profile:")
    for col, val in example_country.iloc[0].items():
        print(f"  {col}: {val}")
    
    print(f"\nPredicted Carbon Emissions:")
    print(f"  Random Forest: {rf_prediction:.2f} tons per capita")
    print(f"  Neural Network: {nn_prediction:.2f} tons per capita")
    print(f"  Average: {(rf_prediction + nn_prediction)/2:.2f} tons per capita")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nKey Insights:")
    print("✓ Models successfully predict carbon emissions")
    print("✓ GDP, industrial output, and energy consumption are key drivers")
    print("✓ Renewable energy and forest coverage reduce emissions")
    print("✓ Can be used for policy planning and climate action")


if __name__ == "__main__":
    main()
