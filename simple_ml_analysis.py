#!/usr/bin/env python3
"""
Simple Bus Performance Analysis (No ML Dependencies Required)
Uses only pandas, numpy, and matplotlib for analysis

Author: Data Analysis Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import Dict, List

# Import our existing analyzer
from eda import MTAACEAnalyzer, CongestionPricingAnalyzer

class SimpleBusAnalyzer:
    """
    Simple statistical analysis without ML dependencies
    """

    def __init__(self):
        self.ace_analyzer = MTAACEAnalyzer()
        self.cp_analyzer = CongestionPricingAnalyzer(self.ace_analyzer)

    def create_comprehensive_dataset(self, sample_size: int = 25000) -> pd.DataFrame:
        """
        Create comprehensive dataset for analysis
        """
        print("Creating comprehensive analysis dataset...")

        # Get ACE violations
        print("1. Fetching ACE violations...")
        violations = self.ace_analyzer.fetch_multiple_batches(
            total_records=sample_size,
            order_by="first_occurrence DESC"
        )

        if violations.empty:
            return pd.DataFrame()

        # Get speed data
        print("2. Fetching speed data...")
        speed_analysis = self.cp_analyzer.analyze_cbd_bus_speeds()

        # Add features
        print("3. Engineering features...")
        df = self.add_analysis_features(violations, speed_analysis)

        return df

    def add_analysis_features(self, violations: pd.DataFrame, speed_analysis: Dict) -> pd.DataFrame:
        """
        Add analysis features to the dataset
        """
        df = violations.copy()

        # Time features
        df['hour'] = df['first_occurrence'].dt.hour
        df['day_of_week'] = df['first_occurrence'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))).astype(int)

        # Route features
        df['is_cuny_route'] = df['bus_route_id'].isin(self.cp_analyzer.cbd_routes).astype(int)
        df['is_express_route'] = df['bus_route_id'].str.contains(r'\+', na=False).astype(int)

        # Add speed data if available
        if speed_analysis and 'bus_speeds' in speed_analysis:
            speed_data = speed_analysis['bus_speeds'].get('route_analysis', {})
            route_speeds = {}
            for route, data in speed_data.items():
                if 'speed_stats' in data and data['speed_stats']:
                    route_speeds[route] = data['speed_stats'].get('mean', 6.5)

            df['route_avg_speed'] = df['bus_route_id'].map(route_speeds).fillna(6.5)
            df['speed_category'] = pd.cut(df['route_avg_speed'],
                                        bins=[0, 6, 8, 12, 100],
                                        labels=['slow', 'moderate', 'fast', 'express'])

        # Calculate delay impact score
        df['delay_impact_score'] = self.calculate_delay_impact(df)

        return df

    def calculate_delay_impact(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate delay impact score based on violation patterns
        """
        # Base impact from violation type
        impact_map = {
            'MOBILE BUS STOP': 3.0,
            'MOBILE BUS LANE': 2.5,
            'MOBILE DOUBLE PARKED': 2.0
        }

        base_impact = df['violation_type'].map(impact_map).fillna(1.0)

        # Rush hour multiplier
        impact = base_impact * (1 + 0.5 * df['is_rush_hour'])

        # CUNY route multiplier (student impact)
        impact = impact * (1 + 0.3 * df['is_cuny_route'])

        return impact

    def analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze temporal patterns in violations
        """
        print("Analyzing temporal patterns...")

        patterns = {}

        # Hourly analysis
        hourly_violations = df.groupby('hour').agg({
            'violation_id': 'count',
            'delay_impact_score': 'mean',
            'is_cuny_route': 'sum'
        }).round(2)

        patterns['hourly'] = {
            'peak_hour': int(hourly_violations['violation_id'].idxmax()),
            'peak_violations': int(hourly_violations['violation_id'].max()),
            'avg_impact_by_hour': hourly_violations['delay_impact_score'].to_dict()
        }

        # Day of week analysis
        daily_violations = df.groupby('day_of_week').agg({
            'violation_id': 'count',
            'delay_impact_score': 'mean'
        }).round(2)

        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        patterns['daily'] = {
            'violations_by_day': dict(zip(day_names, daily_violations['violation_id'].values)),
            'impact_by_day': dict(zip(day_names, daily_violations['delay_impact_score'].values))
        }

        # Rush hour vs non-rush hour
        rush_hour_stats = df.groupby('is_rush_hour').agg({
            'violation_id': 'count',
            'delay_impact_score': 'mean',
            'route_avg_speed': 'mean'
        }).round(2)

        patterns['rush_hour_impact'] = {
            'rush_hour_violations': int(rush_hour_stats.loc[1, 'violation_id']) if 1 in rush_hour_stats.index else 0,
            'non_rush_violations': int(rush_hour_stats.loc[0, 'violation_id']) if 0 in rush_hour_stats.index else 0,
            'rush_hour_avg_impact': float(rush_hour_stats.loc[1, 'delay_impact_score']) if 1 in rush_hour_stats.index else 0,
            'rush_hour_avg_speed': float(rush_hour_stats.loc[1, 'route_avg_speed']) if 1 in rush_hour_stats.index else 0
        }

        return patterns

    def analyze_route_performance(self, df: pd.DataFrame) -> Dict:
        """
        Analyze performance by route
        """
        print("Analyzing route performance...")

        route_analysis = df.groupby('bus_route_id').agg({
            'violation_id': 'count',
            'delay_impact_score': ['mean', 'std'],
            'route_avg_speed': 'first',
            'is_cuny_route': 'first',
            'is_rush_hour': 'mean'
        }).round(2)

        route_analysis.columns = ['violation_count', 'avg_impact', 'impact_variability',
                                'avg_speed', 'is_cuny', 'rush_hour_rate']
        route_analysis = route_analysis.reset_index()

        # Identify problem routes
        problem_routes = route_analysis[
            (route_analysis['avg_impact'] > route_analysis['avg_impact'].quantile(0.75)) |
            (route_analysis['avg_speed'] < 7) |
            (route_analysis['violation_count'] > route_analysis['violation_count'].quantile(0.75))
        ].sort_values('avg_impact', ascending=False)

        # CUNY route specific analysis
        cuny_routes = route_analysis[route_analysis['is_cuny'] == 1].sort_values('avg_impact', ascending=False)

        return {
            'all_routes': route_analysis.to_dict('records'),
            'problem_routes': problem_routes.to_dict('records'),
            'cuny_routes': cuny_routes.to_dict('records'),
            'summary_stats': {
                'total_routes': len(route_analysis),
                'cuny_routes_count': len(cuny_routes),
                'problem_routes_count': len(problem_routes),
                'avg_speed_all_routes': float(route_analysis['avg_speed'].mean()),
                'avg_impact_all_routes': float(route_analysis['avg_impact'].mean())
            }
        }

    def predict_high_impact_periods(self, df: pd.DataFrame) -> Dict:
        """
        Predict high impact periods using simple statistical methods
        """
        print("Predicting high impact periods...")

        # Calculate statistical thresholds
        high_impact_threshold = df['delay_impact_score'].quantile(0.8)
        high_violation_threshold = df.groupby(['hour', 'day_of_week']).size().quantile(0.8)

        # Identify high-risk time periods
        time_impact = df.groupby(['hour', 'day_of_week']).agg({
            'delay_impact_score': 'mean',
            'violation_id': 'count'
        }).reset_index()

        high_risk_periods = time_impact[
            (time_impact['delay_impact_score'] > high_impact_threshold) |
            (time_impact['violation_id'] > high_violation_threshold)
        ]

        # Create recommendations
        recommendations = []
        for _, period in high_risk_periods.iterrows():
            day_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][int(period['day_of_week'])]
            recommendations.append({
                'time_period': f"{day_name} {int(period['hour']):02d}:00",
                'predicted_impact': round(float(period['delay_impact_score']), 2),
                'expected_violations': int(period['violation_id']),
                'recommendation': f"Increase enforcement during {day_name} {int(period['hour']):02d}:00"
            })

        return {
            'high_impact_threshold': round(high_impact_threshold, 2),
            'high_risk_periods': sorted(recommendations, key=lambda x: x['predicted_impact'], reverse=True),
            'total_high_risk_periods': len(recommendations)
        }

    def create_visualizations(self, df: pd.DataFrame, save_plots: bool = True):
        """
        Create comprehensive visualizations
        """
        print("Creating visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Bus Performance Statistical Analysis', fontsize=16)

        # 1. Violations by hour
        hourly_data = df.groupby('hour')['violation_id'].count()
        axes[0, 0].bar(hourly_data.index, hourly_data.values, color='skyblue')
        axes[0, 0].set_title('Violations by Hour of Day')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Number of Violations')

        # 2. Impact score by route (top 10)
        route_impact = df.groupby('bus_route_id')['delay_impact_score'].mean().sort_values(ascending=False).head(10)
        axes[0, 1].barh(route_impact.index, route_impact.values, color='lightcoral')
        axes[0, 1].set_title('Top 10 Routes by Delay Impact')
        axes[0, 1].set_xlabel('Average Impact Score')

        # 3. Speed vs violations scatter
        if 'route_avg_speed' in df.columns:
            route_summary = df.groupby('bus_route_id').agg({
                'route_avg_speed': 'first',
                'violation_id': 'count'
            }).reset_index()

            scatter = axes[0, 2].scatter(route_summary['route_avg_speed'],
                                       route_summary['violation_id'],
                                       alpha=0.6, color='green')
            axes[0, 2].set_title('Route Speed vs Violations')
            axes[0, 2].set_xlabel('Average Speed (mph)')
            axes[0, 2].set_ylabel('Violation Count')

        # 4. Day of week analysis
        daily_data = df.groupby('day_of_week')['violation_id'].count()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1, 0].bar(day_names, daily_data.values, color='orange')
        axes[1, 0].set_title('Violations by Day of Week')
        axes[1, 0].set_ylabel('Number of Violations')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 5. CUNY vs Non-CUNY routes
        cuny_comparison = df.groupby('is_cuny_route').agg({
            'delay_impact_score': 'mean',
            'violation_id': 'count'
        })

        categories = ['Non-CUNY Routes', 'CUNY Routes']
        axes[1, 1].bar(categories, cuny_comparison['delay_impact_score'].values, color=['gray', 'purple'])
        axes[1, 1].set_title('Impact Score: CUNY vs Non-CUNY Routes')
        axes[1, 1].set_ylabel('Average Impact Score')

        # 6. Rush hour comparison
        rush_comparison = df.groupby('is_rush_hour').agg({
            'delay_impact_score': 'mean',
            'violation_id': 'count'
        })

        rush_labels = ['Non-Rush Hours', 'Rush Hours']
        axes[1, 2].bar(rush_labels, rush_comparison['delay_impact_score'].values, color=['lightblue', 'red'])
        axes[1, 2].set_title('Impact Score: Rush vs Non-Rush Hours')
        axes[1, 2].set_ylabel('Average Impact Score')

        plt.tight_layout()

        if save_plots:
            plt.savefig('simple_bus_analysis.png', dpi=300, bbox_inches='tight')
            print("Visualizations saved to 'simple_bus_analysis.png'")

        plt.show()

    def generate_comprehensive_report(self) -> Dict:
        """
        Generate comprehensive statistical analysis report
        """
        print("=== SIMPLE BUS PERFORMANCE ANALYSIS ===")

        # Create dataset
        df = self.create_comprehensive_dataset(sample_size=30000)

        if df.empty:
            return {"error": "No data available"}

        # Run analyses
        temporal_analysis = self.analyze_temporal_patterns(df)
        route_analysis = self.analyze_route_performance(df)
        predictions = self.predict_high_impact_periods(df)

        # Create visualizations
        self.create_visualizations(df)

        # Generate insights
        insights = self.generate_insights(temporal_analysis, route_analysis, predictions)

        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'dataset_size': len(df),
            'temporal_patterns': temporal_analysis,
            'route_performance': route_analysis,
            'predictions': predictions,
            'key_insights': insights
        }

        return report

    def generate_insights(self, temporal: Dict, routes: Dict, predictions: Dict) -> List[str]:
        """
        Generate actionable insights
        """
        insights = []

        # Temporal insights
        peak_hour = temporal['hourly']['peak_hour']
        insights.append(f"Peak violation hour: {peak_hour}:00 with {temporal['hourly']['peak_violations']} violations")

        # Route insights
        worst_cuny_route = routes['cuny_routes'][0] if routes['cuny_routes'] else None
        if worst_cuny_route:
            insights.append(f"Highest impact CUNY route: {worst_cuny_route['bus_route_id']} (impact: {worst_cuny_route['avg_impact']})")

        # Speed insights
        avg_speed = routes['summary_stats']['avg_speed_all_routes']
        if avg_speed < 8:
            insights.append(f"Critically low average speed: {avg_speed:.1f} mph across all routes")

        # Prediction insights
        high_risk_count = predictions['total_high_risk_periods']
        insights.append(f"Identified {high_risk_count} high-risk time periods requiring attention")

        # Actionable recommendations
        insights.extend([
            "Focus enforcement during identified peak hours",
            "Prioritize improvement on high-impact CUNY routes",
            "Consider dynamic scheduling based on predicted patterns",
            "Implement real-time monitoring for problem routes"
        ])

        return insights


def main():
    """
    Main function to run simple analysis
    """
    print("=== SIMPLE BUS PERFORMANCE ANALYSIS ===")

    analyzer = SimpleBusAnalyzer()
    report = analyzer.generate_comprehensive_report()

    # Save report
    with open('simple_bus_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print("\n=== ANALYSIS COMPLETE ===")
    print("Files created:")
    print("- simple_bus_analysis_report.json")
    print("- simple_bus_analysis.png")

    # Print key insights
    if 'key_insights' in report:
        print("\n=== KEY INSIGHTS ===")
        for insight in report['key_insights']:
            print(f"• {insight}")

    return report


if __name__ == "__main__":
    main()