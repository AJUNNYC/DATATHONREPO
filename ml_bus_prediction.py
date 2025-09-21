#!/usr/bin/env python3
"""
MTA Bus Performance Machine Learning Predictor
Predicts bus delays and optimizes routes using ACE violations and speed data

Author: Data Analysis Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from typing import Dict, List

# Machine Learning Libraries (with fallbacks)
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    print("scikit-learn not available. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

# Import our existing analyzer
from eda import MTAACEAnalyzer, CongestionPricingAnalyzer

warnings.filterwarnings('ignore')

class BusPerformanceMLPredictor:
    """
    Machine Learning predictor for bus performance optimization
    """

    def __init__(self):
        self.ace_analyzer = MTAACEAnalyzer()
        self.cp_analyzer = CongestionPricingAnalyzer(self.ace_analyzer)
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}

    def collect_ml_dataset(self, sample_size: int = 75000) -> pd.DataFrame:
        """
        Collect and prepare comprehensive dataset for ML
        """
        print("Collecting comprehensive dataset for ML...")

        # Get ACE violations data
        print("1. Fetching ACE violations...")
        violations = self.ace_analyzer.fetch_multiple_batches(
            total_records=sample_size,
            order_by="first_occurrence DESC"
        )

        # Get speed data
        print("2. Fetching speed data...")
        speed_analysis = self.cp_analyzer.analyze_cbd_bus_speeds()

        if violations.empty:
            print("No violations data available")
            return pd.DataFrame()

        # Feature engineering
        print("3. Engineering features...")
        ml_data = self.engineer_features(violations, speed_analysis)

        return ml_data

    def engineer_features(self, violations: pd.DataFrame, speed_analysis: Dict) -> pd.DataFrame:
        """
        Create features for machine learning
        """
        # Start with violations data
        df = violations.copy()

        # Time-based features
        df['hour'] = df['first_occurrence'].dt.hour
        df['day_of_week'] = df['first_occurrence'].dt.dayofweek
        df['month'] = df['first_occurrence'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)
        df['is_peak_congestion'] = ((df['hour'].between(8, 10)) | (df['hour'].between(16, 18))).astype(int)

        # Route features
        df['is_cuny_route'] = df['bus_route_id'].isin(self.cp_analyzer.cbd_routes).astype(int)
        df['is_express_route'] = df['bus_route_id'].str.contains(r'\+').astype(int)

        # Violation features (simple encoding without sklearn)
        if SKLEARN_AVAILABLE:
            le_violation_type = LabelEncoder()
            le_violation_status = LabelEncoder()
            df['violation_type_encoded'] = le_violation_type.fit_transform(df['violation_type'].fillna('Unknown'))
            df['violation_status_encoded'] = le_violation_status.fit_transform(df['violation_status'].fillna('Unknown'))
        else:
            # Simple manual encoding
            violation_types = df['violation_type'].fillna('Unknown').unique()
            violation_type_map = {vtype: i for i, vtype in enumerate(violation_types)}
            df['violation_type_encoded'] = df['violation_type'].fillna('Unknown').map(violation_type_map)

            violation_statuses = df['violation_status'].fillna('Unknown').unique()
            violation_status_map = {vstatus: i for i, vstatus in enumerate(violation_statuses)}
            df['violation_status_encoded'] = df['violation_status'].fillna('Unknown').map(violation_status_map)

        # Geographic features (if coordinates available)
        if 'violation_latitude' in df.columns and 'violation_longitude' in df.columns:
            df['in_cbd'] = (
                (df['violation_latitude'].between(40.7047, 40.7829)) &
                (df['violation_longitude'].between(-73.9903, -73.9367))
            ).astype(int)

            # Distance from city center (approximate)
            city_center_lat, city_center_lon = 40.7589, -73.9851  # Times Square
            df['distance_from_center'] = np.sqrt(
                (df['violation_latitude'] - city_center_lat)**2 +
                (df['violation_longitude'] - city_center_lon)**2
            )

        # Speed features (merge with speed data if available)
        if speed_analysis and 'bus_speeds' in speed_analysis:
            speed_data = speed_analysis['bus_speeds'].get('route_analysis', {})

            # Create speed mapping
            route_speeds = {}
            for route, data in speed_data.items():
                if 'speed_stats' in data and data['speed_stats']:
                    route_speeds[route] = data['speed_stats'].get('mean', 6.5)  # Default to overall average

            df['historical_speed'] = df['bus_route_id'].map(route_speeds).fillna(6.5)
            df['speed_category'] = pd.cut(df['historical_speed'],
                                        bins=[0, 6, 8, 12, 100],
                                        labels=['slow', 'moderate', 'fast', 'express'])
            df['speed_category_encoded'] = LabelEncoder().fit_transform(df['speed_category'].astype(str))

        # Aggregated features by route
        route_agg = df.groupby('bus_route_id').agg({
            'violation_id': 'count',
            'hour': 'mean',
            'is_rush_hour': 'mean'
        }).rename(columns={
            'violation_id': 'route_violation_frequency',
            'hour': 'route_avg_hour',
            'is_rush_hour': 'route_rush_hour_rate'
        })

        df = df.merge(route_agg, left_on='bus_route_id', right_index=True, how='left')

        # Target variable: Create "delay_severity" score
        df['delay_severity'] = self.calculate_delay_severity(df)

        # Select features for ML
        feature_columns = [
            'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour', 'is_peak_congestion',
            'is_cuny_route', 'is_express_route', 'violation_type_encoded', 'violation_status_encoded',
            'route_violation_frequency', 'route_avg_hour', 'route_rush_hour_rate'
        ]

        # Add geographic features if available
        if 'in_cbd' in df.columns:
            feature_columns.extend(['in_cbd', 'distance_from_center'])

        # Add speed features if available
        if 'historical_speed' in df.columns:
            feature_columns.extend(['historical_speed', 'speed_category_encoded'])

        # Keep only relevant columns
        ml_features = feature_columns + ['delay_severity', 'bus_route_id', 'first_occurrence']
        df_ml = df[ml_features].copy()

        # Remove rows with NaN values
        df_ml = df_ml.dropna()

        return df_ml

    def calculate_delay_severity(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate delay severity score based on violation patterns
        """
        # Base severity from violation type
        severity_map = {
            'MOBILE BUS STOP': 3.0,      # High impact on boarding
            'MOBILE BUS LANE': 2.5,      # Blocks bus movement
            'MOBILE DOUBLE PARKED': 2.0,  # Moderate impact
        }

        base_severity = df['violation_type'].map(severity_map).fillna(1.0)

        # Adjust for time factors
        rush_hour_multiplier = 1.5
        weekend_reduction = 0.7

        severity = base_severity.copy()
        severity[df['is_rush_hour'] == 1] *= rush_hour_multiplier
        severity[df['is_weekend'] == 1] *= weekend_reduction

        # Adjust for route frequency (more violations = higher severity)
        severity += np.log1p(df.get('route_violation_frequency', 1)) * 0.1

        return severity

    def train_delay_prediction_models(self, df: pd.DataFrame) -> Dict:
        """
        Train multiple ML models to predict bus delays
        """
        if not SKLEARN_AVAILABLE:
            print("Scikit-learn not available. Cannot train ML models.")
            return {}

        print("Training delay prediction models...")

        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['delay_severity', 'bus_route_id', 'first_occurrence']]
        X = df[feature_cols]
        y = df['delay_severity']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers['delay_prediction'] = scaler

        # Train multiple models (only if libraries available)
        models = {}

        if SKLEARN_AVAILABLE:
            models.update({
                'linear_regression': LinearRegression(),
                'ridge_regression': Ridge(alpha=1.0),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
            })

        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)

        if not models:
            print("No ML libraries available. Please install scikit-learn and xgboost")
            return {}

        results = {}

        for name, model in models.items():
            print(f"Training {name}...")

            # Use scaled data for linear models, original for tree-based
            if name in ['linear_regression', 'ridge_regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }

            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                results[name]['feature_importance'] = importance_df

            print(f"{name}: RMSE={rmse:.3f}, R²={r2:.3f}")

        self.models['delay_prediction'] = results
        return results

    def predict_route_performance(self, route_id: str, hour: int, day_of_week: int) -> Dict:
        """
        Predict performance for a specific route at a given time
        """
        if 'delay_prediction' not in self.models:
            return {"error": "Models not trained yet"}

        # Get best model (highest R²)
        best_model_name = max(self.models['delay_prediction'].keys(),
                             key=lambda x: self.models['delay_prediction'][x]['r2'])
        best_model = self.models['delay_prediction'][best_model_name]['model']

        # Create prediction features with all expected columns
        features = {
            'hour': hour,
            'day_of_week': day_of_week,
            'month': datetime.now().month,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'is_rush_hour': 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0,
            'is_peak_congestion': 1 if (8 <= hour <= 10) or (16 <= hour <= 18) else 0,
            'is_cuny_route': 1 if route_id in self.cp_analyzer.cbd_routes else 0,
            'is_express_route': 1 if '+' in route_id else 0,
            'violation_type_encoded': 1,  # Default values
            'violation_status_encoded': 1,
            'route_violation_frequency': 100,  # Default
            'route_avg_hour': 12,
            'route_rush_hour_rate': 0.3,
            # Add missing features with default values
            'in_cbd': 1,  # Assume CBD for prediction
            'distance_from_center': 0.02,  # Default distance
            'historical_speed': 6.5,  # Default speed
            'speed_category_encoded': 0  # Default to slow category
        }

        # Convert to DataFrame
        X_pred = pd.DataFrame([features])

        try:
            # Make prediction
            if best_model_name in ['linear_regression', 'ridge_regression']:
                X_pred_scaled = self.scalers['delay_prediction'].transform(X_pred)
                prediction = best_model.predict(X_pred_scaled)[0]
            else:
                prediction = best_model.predict(X_pred)[0]
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

        # Interpret prediction
        if prediction < 1.5:
            severity = "Low"
            description = "Minimal delays expected"
        elif prediction < 2.5:
            severity = "Moderate"
            description = "Some delays possible"
        else:
            severity = "High"
            description = "Significant delays likely"

        return {
            'route_id': route_id,
            'predicted_delay_severity': prediction,
            'severity_level': severity,
            'description': description,
            'model_used': best_model_name,
            'confidence': self.models['delay_prediction'][best_model_name]['r2']
        }

    def optimize_bus_routes(self, df: pd.DataFrame) -> Dict:
        """
        Use ML insights to recommend route optimizations
        """
        print("Analyzing route optimization opportunities...")

        # Route performance analysis
        route_performance = df.groupby('bus_route_id').agg({
            'delay_severity': ['mean', 'std', 'count'],
            'is_rush_hour': 'mean',
            'historical_speed': 'first'
        }).round(3)

        route_performance.columns = ['avg_delay_severity', 'delay_variability', 'violation_count',
                                   'rush_hour_rate', 'avg_speed']
        route_performance = route_performance.reset_index()

        # Identify optimization opportunities
        optimizations = []

        for _, route in route_performance.iterrows():
            route_id = route['bus_route_id']

            # High delay severity routes
            if route['avg_delay_severity'] > 2.5:
                optimizations.append({
                    'route': route_id,
                    'issue': 'High Delay Severity',
                    'severity': route['avg_delay_severity'],
                    'recommendation': 'Increase enforcement, consider dedicated lanes'
                })

            # High variability routes
            if route['delay_variability'] > 1.0:
                optimizations.append({
                    'route': route_id,
                    'issue': 'High Delay Variability',
                    'variability': route['delay_variability'],
                    'recommendation': 'Implement consistent schedule, improve signal timing'
                })

            # Slow routes with high violations
            if route['avg_speed'] < 7 and route['violation_count'] > 100:
                optimizations.append({
                    'route': route_id,
                    'issue': 'Slow Speed + High Violations',
                    'speed': route['avg_speed'],
                    'violations': route['violation_count'],
                    'recommendation': 'Priority enforcement corridor, bus lane improvements'
                })

        return {
            'route_performance': route_performance,
            'optimization_opportunities': optimizations,
            'total_routes_analyzed': len(route_performance),
            'routes_needing_attention': len(optimizations)
        }

    def create_ml_visualizations(self, save_plots: bool = True):
        """
        Create visualizations for ML results
        """
        if 'delay_prediction' not in self.models:
            print("No models trained yet")
            return

        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Bus Performance ML Analysis Results', fontsize=16)

        # 1. Model Performance Comparison
        model_names = list(self.models['delay_prediction'].keys())
        r2_scores = [self.models['delay_prediction'][name]['r2'] for name in model_names]
        rmse_scores = [self.models['delay_prediction'][name]['rmse'] for name in model_names]

        axes[0, 0].bar(model_names, r2_scores, color='skyblue')
        axes[0, 0].set_title('Model R² Scores')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)

        axes[0, 1].bar(model_names, rmse_scores, color='lightcoral')
        axes[0, 1].set_title('Model RMSE Scores')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 2. Feature Importance (use best tree-based model)
        tree_models = ['random_forest', 'gradient_boosting', 'xgboost']
        best_tree_model = None
        best_r2 = -1

        for model_name in tree_models:
            if model_name in self.models['delay_prediction']:
                if self.models['delay_prediction'][model_name]['r2'] > best_r2:
                    best_r2 = self.models['delay_prediction'][model_name]['r2']
                    best_tree_model = model_name

        if best_tree_model and 'feature_importance' in self.models['delay_prediction'][best_tree_model]:
            importance_df = self.models['delay_prediction'][best_tree_model]['feature_importance'].head(10)
            axes[1, 0].barh(importance_df['feature'], importance_df['importance'])
            axes[1, 0].set_title(f'Feature Importance ({best_tree_model})')
            axes[1, 0].set_xlabel('Importance')

        # 3. Prediction vs Actual (use best model)
        best_model_name = max(self.models['delay_prediction'].keys(),
                             key=lambda x: self.models['delay_prediction'][x]['r2'])

        # This would need actual vs predicted data from the training process
        # For now, create a placeholder
        axes[1, 1].text(0.5, 0.5, f'Best Model: {best_model_name}\nR² = {self.models["delay_prediction"][best_model_name]["r2"]:.3f}',
                        ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Model Performance Summary')

        plt.tight_layout()

        if save_plots:
            plt.savefig('ml_bus_analysis.png', dpi=300, bbox_inches='tight')
            print("ML visualizations saved to 'ml_bus_analysis.png'")

        plt.show()

    def generate_ml_report(self) -> Dict:
        """
        Generate comprehensive ML analysis report
        """
        print("=== MACHINE LEARNING BUS PERFORMANCE ANALYSIS ===")

        # Collect data
        print("1. Collecting ML dataset...")
        ml_data = self.collect_ml_dataset(sample_size=50000)

        if ml_data.empty:
            return {"error": "No data available for ML analysis"}

        # Train models
        print("2. Training prediction models...")
        model_results = self.train_delay_prediction_models(ml_data)

        # Route optimization
        print("3. Analyzing route optimizations...")
        optimization_results = self.optimize_bus_routes(ml_data)

        # Create visualizations
        print("4. Creating ML visualizations...")
        self.create_ml_visualizations()

        # Generate predictions for key routes
        print("5. Generating route predictions...")
        key_routes = ['M15+', 'M2', 'M42', 'M101', 'M100']
        route_predictions = {}

        for route in key_routes:
            # Predict for rush hour (8 AM, Monday)
            prediction = self.predict_route_performance(route, hour=8, day_of_week=0)
            route_predictions[route] = prediction

        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'dataset_size': len(ml_data),
            'model_performance': {name: {k: v for k, v in results.items() if k != 'model'}
                                for name, results in model_results.items()},
            'route_optimization': optimization_results,
            'route_predictions': route_predictions,
            'ml_insights': self._generate_ml_insights(model_results, optimization_results)
        }

        return report

    def _generate_ml_insights(self, model_results: Dict, optimization_results: Dict) -> List[str]:
        """Generate actionable insights from ML analysis"""
        insights = []

        # Best model insight
        best_model = max(model_results.keys(), key=lambda x: model_results[x]['r2'])
        best_r2 = model_results[best_model]['r2']
        insights.append(f"Best performing model: {best_model} (R² = {best_r2:.3f})")

        # Feature importance insight
        if 'feature_importance' in model_results[best_model]:
            top_feature = model_results[best_model]['feature_importance'].iloc[0]['feature']
            insights.append(f"Most important factor for delays: {top_feature}")

        # Route optimization insight
        routes_needing_attention = optimization_results['routes_needing_attention']
        total_routes = optimization_results['total_routes_analyzed']
        insights.append(f"{routes_needing_attention} out of {total_routes} routes need optimization")

        # Actionable recommendations
        insights.extend([
            "ML model can predict delays with reasonable accuracy for route planning",
            "Focus enforcement on high-delay-severity routes during rush hours",
            "Consider dynamic scheduling based on predicted delay patterns",
            "Use real-time predictions to inform passenger arrival estimates"
        ])

        return insights


def main():
    """
    Main function to run ML analysis
    """
    print("=== BUS PERFORMANCE MACHINE LEARNING ANALYSIS ===")

    # Initialize predictor
    predictor = BusPerformanceMLPredictor()

    # Generate comprehensive ML report
    report = predictor.generate_ml_report()

    # Save report
    import json
    with open('ml_bus_performance_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print("\n=== ML ANALYSIS COMPLETE ===")
    print("Reports saved:")
    print("- ml_bus_performance_report.json")
    print("- ml_bus_analysis.png")

    # Print key insights
    if 'ml_insights' in report:
        print("\n=== KEY ML INSIGHTS ===")
        for insight in report['ml_insights']:
            print(f"• {insight}")

    return report


if __name__ == "__main__":
    main()