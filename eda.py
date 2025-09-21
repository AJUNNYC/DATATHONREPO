#!/usr/bin/env python3
"""
MTA Bus Automated Camera Enforcement (ACE) Exploratory Data Analysis
Author: Data Analysis Team
Date: September 2025

This script provides comprehensive EDA for the MTA ACE dataset without requiring
local storage of the full dataset. It uses the NYC Open Data API to fetch data
in chunks and perform analysis focused on CUNY student transportation patterns.
"""

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import json
from collections import Counter
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class MTAACEAnalyzer:
    """
    Comprehensive analyzer for MTA Bus Automated Camera Enforcement data
    """
    
    def __init__(self, api_base_url: str = "https://data.ny.gov/resource/kh8p-hcbm.json"):
        """
        Initialize the analyzer with API configuration
        
        Args:
            api_base_url: Base URL for the NYC Open Data API
        """
        self.api_base_url = api_base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MTA-ACE-Analyzer/1.0',
            'Accept': 'application/json'
        })
        
        # CUNY-relevant bus routes based on campus locations
        self.cuny_routes = {
            'Manhattan': ['M2', 'M4', 'M5', 'M11', 'M15+', 'M15', 'M60+', 'M60', 'M100', 'M101', 'M104', 'M42', 'M34+'],
            'Brooklyn': ['B35', 'B41', 'B44+', 'B46+', 'B25', 'B26', 'B62', 'B82+'],
            'Queens': ['Q43', 'Q44+', 'Q53+', 'Q58', 'Q69', 'Q70', 'Q5'],
            'Bronx': ['BX6+', 'BX12+', 'BX19', 'BX28', 'BX35', 'BX36', 'BX38', 'BX41+'],
            'Staten_Island': ['S53', 'S61', 'S62', 'S92', 'S93']
        }
        
        # Central Business District boundaries (approximately)
        self.cbd_bounds = {
            'north': 40.7829,  # ~60th Street
            'south': 40.7047,  # ~Battery Park
            'east': -73.9367,  # ~East River
            'west': -73.9903   # ~Hudson River
        }
        
        # Congestion pricing implementation date
        self.congestion_pricing_start = datetime(2025, 1, 5)
        
    def fetch_data_batch(self, limit: int = 1000, offset: int = 0,
                        where_clause: str = None, order_by: str = None) -> pd.DataFrame:
        """
        Fetch a batch of data from the API
        
        Args:
            limit: Number of records to fetch
            offset: Starting offset for pagination
            where_clause: SQL-like WHERE clause for filtering
            order_by: Field to order results by
            
        Returns:
            DataFrame with the fetched data
        """
        params = {
            '$limit': limit,
            '$offset': offset
        }
        
        if where_clause:
            params['$where'] = where_clause
        if order_by:
            params['$order'] = order_by
            
        try:
            response = self.session.get(self.api_base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            
            # Convert datetime columns
            if 'first_occurrence' in df.columns:
                df['first_occurrence'] = pd.to_datetime(df['first_occurrence'])
            if 'last_occurrence' in df.columns:
                df['last_occurrence'] = pd.to_datetime(df['last_occurrence'])
                
            # Convert numeric columns
            numeric_cols = ['violation_latitude', 'violation_longitude', 
                          'bus_stop_latitude', 'bus_stop_longitude']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            return df
            
        except requests.RequestException as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def fetch_multiple_batches(self, total_records: int, where_clause: str = None,
                              order_by: str = None, progress_interval: int = 5) -> pd.DataFrame:
        """
        Fetch multiple batches of data to get more than 1000 records

        Args:
            total_records: Total number of records to fetch
            where_clause: SQL-like WHERE clause for filtering
            order_by: Field to order results by
            progress_interval: Show progress every N batches

        Returns:
            Combined DataFrame with all fetched data
        """
        all_data = []
        batch_size = 1000  # API limit
        batch_count = 0
        total_batches = (total_records + batch_size - 1) // batch_size

        print(f"Fetching {total_records:,} records in {total_batches} batches...")

        for offset in range(0, total_records, batch_size):
            current_limit = min(batch_size, total_records - offset)
            batch_count += 1

            # Show progress at intervals
            if batch_count % progress_interval == 0 or batch_count == total_batches:
                print(f"Progress: {batch_count}/{total_batches} batches ({len(all_data)*batch_size if all_data else 0:,} records)")

            batch_df = self.fetch_data_batch(
                limit=current_limit,
                offset=offset,
                where_clause=where_clause,
                order_by=order_by
            )

            if batch_df.empty:
                print(f"No more data available at offset {offset}")
                break

            all_data.append(batch_df)

            # If we got fewer records than requested, we've reached the end
            if len(batch_df) < current_limit:
                print(f"Reached end of available data at {len(all_data)*batch_size + len(batch_df):,} records")
                break

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"Successfully fetched {len(combined_df):,} records")
            return combined_df
        else:
            print("No data fetched")
            return pd.DataFrame()

    def fetch_comprehensive_dataset(self, max_records: int = 100000,
                                   where_clause: str = None,
                                   order_by: str = "first_occurrence DESC") -> pd.DataFrame:
        """
        Fetch a comprehensive dataset for thorough analysis

        Args:
            max_records: Maximum number of records to fetch (default 100k)
            where_clause: SQL-like WHERE clause for filtering
            order_by: Field to order results by

        Returns:
            Large DataFrame for comprehensive analysis
        """
        print(f"Fetching comprehensive dataset (up to {max_records:,} records)...")
        return self.fetch_multiple_batches(
            total_records=max_records,
            where_clause=where_clause,
            order_by=order_by,
            progress_interval=10  # Show progress every 10 batches for large datasets
        )

    def get_data_summary(self) -> Dict:
        """
        Get a summary of the dataset without downloading all data
        
        Returns:
            Dictionary with dataset summary statistics
        """
        # Get total count
        count_response = self.session.get(f"{self.api_base_url}?$select=count(*)")
        total_records = int(count_response.json()[0]['count'])
        
        # Get larger sample for basic stats using pagination
        sample_df = self.fetch_multiple_batches(total_records=10000, order_by="first_occurrence DESC")
        
        summary = {
            'total_records': total_records,
            'sample_size': len(sample_df),
            'date_range': {
                'earliest': sample_df['first_occurrence'].min().strftime('%Y-%m-%d'),
                'latest': sample_df['first_occurrence'].max().strftime('%Y-%m-%d')
            },
            'violation_types': sample_df['violation_type'].value_counts().to_dict(),
            'violation_statuses': sample_df['violation_status'].value_counts().to_dict(),
            'unique_routes': sample_df['bus_route_id'].nunique(),
            'unique_vehicles': sample_df['vehicle_id'].nunique()
        }
        
        return summary
    
    def analyze_cuny_routes(self, sample_size: int = 20000) -> pd.DataFrame:
        """
        Analyze violations on CUNY-relevant bus routes
        
        Args:
            sample_size: Number of recent records to analyze
            
        Returns:
            DataFrame with CUNY route analysis
        """
        # Flatten CUNY routes list
        all_cuny_routes = []
        for borough, routes in self.cuny_routes.items():
            all_cuny_routes.extend(routes)
        
        # Create WHERE clause for CUNY routes
        route_filter = " OR ".join([f"bus_route_id = '{route}'" for route in all_cuny_routes])
        where_clause = f"({route_filter})"
        
        # Fetch data for CUNY routes (use pagination if more than 1000 records needed)
        if sample_size > 1000:
            cuny_data = self.fetch_multiple_batches(
                total_records=sample_size,
                where_clause=where_clause,
                order_by="first_occurrence DESC"
            )
        else:
            cuny_data = self.fetch_data_batch(
                limit=sample_size,
                where_clause=where_clause,
                order_by="first_occurrence DESC"
            )
        
        if cuny_data.empty:
            print("No CUNY route data found")
            return pd.DataFrame()
        
        # Add borough classification
        def classify_borough(route_id):
            for borough, routes in self.cuny_routes.items():
                if route_id in routes:
                    return borough
            return 'Other'
        
        cuny_data['borough'] = cuny_data['bus_route_id'].apply(classify_borough)
        
        # Analyze patterns
        route_analysis = cuny_data.groupby(['bus_route_id', 'borough']).agg({
            'violation_id': 'count',
            'violation_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
            'violation_status': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown',
            'first_occurrence': ['min', 'max']
        }).round(2)
        
        route_analysis.columns = ['violation_count', 'most_common_type', 'most_common_status', 
                                'earliest_violation', 'latest_violation']
        route_analysis = route_analysis.reset_index()
        route_analysis = route_analysis.sort_values('violation_count', ascending=False)
        
        return route_analysis
    
    def analyze_exempt_vehicles(self, sample_size: int = 25000) -> Dict:
        """
        Analyze patterns in exempt vehicle violations
        
        Args:
            sample_size: Number of records to analyze
            
        Returns:
            Dictionary with exempt vehicle analysis
        """
        # Fetch data with exempt statuses
        exempt_statuses = ['EXEMPT - BUS/PARATRANSIT', 'EXEMPT - EMERGENCY VEHICLE', 
                          'EXEMPT - COMMERCIAL UNDER 20', 'EXEMPT - OTHER']
        
        exempt_filter = " OR ".join([f"violation_status = '{status}'" for status in exempt_statuses])
        where_clause = f"({exempt_filter})"
        
        # Fetch exempt vehicle data (use pagination if more than 1000 records needed)
        if sample_size > 1000:
            exempt_data = self.fetch_multiple_batches(
                total_records=sample_size,
                where_clause=where_clause,
                order_by="first_occurrence DESC"
            )
        else:
            exempt_data = self.fetch_data_batch(
                limit=sample_size,
                where_clause=where_clause,
                order_by="first_occurrence DESC"
            )
        
        if exempt_data.empty:
            return {}
        
        # Analyze repeat offenders using hashed vehicle IDs
        repeat_offenders = exempt_data['vehicle_id'].value_counts()
        repeat_offenders = repeat_offenders[repeat_offenders > 1]
        
        # Geographic concentration
        location_hotspots = exempt_data.groupby(['stop_name', 'violation_status']).size().reset_index()
        location_hotspots.columns = ['location', 'exempt_type', 'count']
        location_hotspots = location_hotspots.sort_values('count', ascending=False)
        
        # Route analysis
        route_exempt_patterns = exempt_data.groupby(['bus_route_id', 'violation_status']).size().reset_index()
        route_exempt_patterns.columns = ['route', 'exempt_type', 'count']
        route_exempt_patterns = route_exempt_patterns.sort_values('count', ascending=False)
        
        analysis = {
            'total_exempt_violations': len(exempt_data),
            'repeat_offender_count': len(repeat_offenders),
            'most_frequent_repeat_offender': repeat_offenders.iloc[0] if len(repeat_offenders) > 0 else 0,
            'top_violation_locations': location_hotspots.head(10).to_dict('records'),
            'route_patterns': route_exempt_patterns.head(15).to_dict('records'),
            'exempt_status_distribution': exempt_data['violation_status'].value_counts().to_dict()
        }
        
        return analysis
    
    def analyze_congestion_pricing_impact(self) -> Dict:
        """
        Analyze the impact of congestion pricing on ACE violations
        
        Returns:
            Dictionary with before/after analysis
        """
        cp_start = self.congestion_pricing_start
        
        # Define time periods
        before_start = cp_start - timedelta(days=60)  # 2 months before
        after_end = cp_start + timedelta(days=120)    # 4 months after
        
        # Fetch data for CBD routes before congestion pricing
        cbd_routes = ['M15+', 'M15', 'M2', 'M42', 'M34+', 'M4', 'M5', 'M23+']
        route_filter = " OR ".join([f"bus_route_id = '{route}'" for route in cbd_routes])
        
        before_where = f"({route_filter}) AND first_occurrence >= '{before_start.isoformat()}' AND first_occurrence < '{cp_start.isoformat()}'"
        after_where = f"({route_filter}) AND first_occurrence >= '{cp_start.isoformat()}' AND first_occurrence <= '{after_end.isoformat()}'"
        
        # Fetch congestion pricing data (use pagination for larger datasets)
        before_data = self.fetch_multiple_batches(total_records=15000, where_clause=before_where)
        after_data = self.fetch_multiple_batches(total_records=15000, where_clause=after_where)
        
        if before_data.empty or after_data.empty:
            return {"error": "Insufficient data for congestion pricing analysis"}
        
        # Analyze patterns
        before_daily = before_data.groupby(before_data['first_occurrence'].dt.date).size()
        after_daily = after_data.groupby(after_data['first_occurrence'].dt.date).size()
        
        analysis = {
            'before_period': {
                'total_violations': len(before_data),
                'daily_average': before_daily.mean(),
                'most_affected_routes': before_data['bus_route_id'].value_counts().head(5).to_dict()
            },
            'after_period': {
                'total_violations': len(after_data),
                'daily_average': after_daily.mean(),
                'most_affected_routes': after_data['bus_route_id'].value_counts().head(5).to_dict()
            },
            'change_metrics': {
                'violation_change_percent': ((len(after_data) - len(before_data)) / len(before_data)) * 100,
                'daily_average_change': after_daily.mean() - before_daily.mean()
            }
        }
        
        return analysis
    
    def create_geographic_visualization(self, sample_size: int = 10000) -> folium.Map:
        """
        Create an interactive map of violations
        
        Args:
            sample_size: Number of recent violations to plot
            
        Returns:
            Folium map object
        """
        # Fetch recent data with coordinates using pagination for larger datasets
        if sample_size > 1000:
            recent_data = self.fetch_multiple_batches(
                total_records=sample_size,
                order_by="first_occurrence DESC"
            )
        else:
            recent_data = self.fetch_data_batch(
                limit=sample_size,
                order_by="first_occurrence DESC"
            )
        
        # Filter out invalid coordinates
        recent_data = recent_data.dropna(subset=['violation_latitude', 'violation_longitude'])
        recent_data = recent_data[
            (recent_data['violation_latitude'] > 40.4) & 
            (recent_data['violation_latitude'] < 41.0) &
            (recent_data['violation_longitude'] > -74.5) & 
            (recent_data['violation_longitude'] < -73.0)
        ]
        
        if recent_data.empty:
            print("No valid coordinate data for mapping")
            return None
        
        # Create base map centered on NYC
        center_lat = recent_data['violation_latitude'].mean()
        center_lon = recent_data['violation_longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
        
        # Add congestion pricing zone
        cbd_polygon = [
            [self.cbd_bounds['north'], self.cbd_bounds['west']],
            [self.cbd_bounds['north'], self.cbd_bounds['east']],
            [self.cbd_bounds['south'], self.cbd_bounds['east']],
            [self.cbd_bounds['south'], self.cbd_bounds['west']]
        ]
        
        folium.Polygon(
            locations=cbd_polygon,
            color='red',
            weight=2,
            fillColor='red',
            fillOpacity=0.1,
            popup='Congestion Pricing Zone'
        ).add_to(m)
        
        # Color code by violation status
        status_colors = {
            'VIOLATION ISSUED': 'red',
            'EXEMPT - EMERGENCY VEHICLE': 'blue',
            'EXEMPT - BUS/PARATRANSIT': 'green',
            'TECHNICAL ISSUE/OTHER': 'orange',
            'DRIVER/VEHICLE INFO MISSING': 'purple',
            'EXEMPT - OTHER': 'gray'
        }
        
        # Add violation points
        for _, row in recent_data.head(500).iterrows():  # Limit for performance
            color = status_colors.get(row['violation_status'], 'black')
            
            folium.CircleMarker(
                location=[row['violation_latitude'], row['violation_longitude']],
                radius=3,
                popup=f"Route: {row['bus_route_id']}<br>Status: {row['violation_status']}<br>Type: {row['violation_type']}<br>Stop: {row['stop_name']}",
                color=color,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 150px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Violation Status</b></p>
        <p><i class="fa fa-circle" style="color:red"></i> Violation Issued</p>
        <p><i class="fa fa-circle" style="color:blue"></i> Emergency Vehicle</p>
        <p><i class="fa fa-circle" style="color:green"></i> Bus/Paratransit</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Technical Issue</p>
        <p><i class="fa fa-circle" style="color:purple"></i> Missing Info</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def generate_comprehensive_report(self) -> Dict:
        """
        Generate a comprehensive analysis report
        
        Returns:
            Dictionary with all analysis results
        """
        print("Generating comprehensive MTA ACE analysis report...")
        
        # Basic dataset summary
        print("1. Fetching dataset summary...")
        summary = self.get_data_summary()
        
        # CUNY route analysis
        print("2. Analyzing CUNY-relevant routes...")
        cuny_analysis = self.analyze_cuny_routes()
        
        # Exempt vehicle analysis
        print("3. Analyzing exempt vehicles...")
        exempt_analysis = self.analyze_exempt_vehicles()
        
        # Congestion pricing impact
        print("4. Analyzing congestion pricing impact...")
        cp_analysis = self.analyze_congestion_pricing_impact()
        
        # Create visualizations
        print("5. Creating geographic visualization...")
        map_viz = self.create_geographic_visualization()
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'dataset_summary': summary,
            'cuny_route_analysis': cuny_analysis.to_dict('records') if not cuny_analysis.empty else [],
            'exempt_vehicle_analysis': exempt_analysis,
            'congestion_pricing_analysis': cp_analysis,
            'map_created': map_viz is not None
        }
        
        return report
    
    def create_visualizations(self, save_plots: bool = False):
        """
        Create comprehensive visualizations
        
        Args:
            save_plots: Whether to save plots to files
        """
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Get data for visualizations (larger sample sizes using pagination)
        cuny_data = self.analyze_cuny_routes(sample_size=15000)
        exempt_data = self.analyze_exempt_vehicles(sample_size=15000)
        
        if cuny_data.empty:
            print("No data available for visualizations")
            return
        
        # Create subplot figure
        fig = plt.figure(figsize=(20, 15))
        
        # 1. CUNY Routes Violation Count
        plt.subplot(2, 3, 1)
        top_routes = cuny_data.head(10)
        plt.barh(top_routes['bus_route_id'], top_routes['violation_count'])
        plt.title('Top 10 CUNY-Relevant Routes by Violation Count')
        plt.xlabel('Number of Violations')
        plt.tight_layout()
        
        # 2. Borough Distribution
        plt.subplot(2, 3, 2)
        borough_counts = cuny_data.groupby('borough')['violation_count'].sum()
        plt.pie(borough_counts.values, labels=borough_counts.index, autopct='%1.1f%%')
        plt.title('Violations by Borough')
        
        # 3. Violation Types
        plt.subplot(2, 3, 3)
        type_counts = cuny_data['most_common_type'].value_counts()
        plt.bar(type_counts.index, type_counts.values)
        plt.title('Most Common Violation Types')
        plt.xticks(rotation=45)
        
        # 4. Exempt Status Distribution
        if exempt_data and 'exempt_status_distribution' in exempt_data:
            plt.subplot(2, 3, 4)
            exempt_status = exempt_data['exempt_status_distribution']
            plt.pie(exempt_status.values(), labels=exempt_status.keys(), autopct='%1.1f%%')
            plt.title('Exempt Violation Status Distribution')
        
        # 5. Route Patterns for Exempt Vehicles
        if exempt_data and 'route_patterns' in exempt_data:
            plt.subplot(2, 3, 5)
            route_patterns = pd.DataFrame(exempt_data['route_patterns'])
            if not route_patterns.empty:
                top_exempt_routes = route_patterns.head(8)
                plt.barh(top_exempt_routes['route'], top_exempt_routes['count'])
                plt.title('Top Routes for Exempt Vehicle Violations')
                plt.xlabel('Number of Violations')
        
        # 6. Time Series (if data available)
        plt.subplot(2, 3, 6)
        plt.text(0.5, 0.5, 'Time Series Analysis\n(Requires larger dataset)', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Temporal Patterns')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('mta_ace_analysis.png', dpi=300, bbox_inches='tight')
            print("Visualizations saved to 'mta_ace_analysis.png'")
        
        plt.show()


# Example usage and main execution
def main():
    """
    Main function demonstrating how to use the MTAACEAnalyzer
    """
    print("=== MTA Bus ACE Exploratory Data Analysis ===")
    print("Initializing analyzer...")
    
    # Initialize analyzer
    analyzer = MTAACEAnalyzer()
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()
    
    # Print key findings
    print("\n=== KEY FINDINGS ===")
    print(f"Total records in dataset: {report['dataset_summary']['total_records']:,}")
    print(f"Date range: {report['dataset_summary']['date_range']['earliest']} to {report['dataset_summary']['date_range']['latest']}")
    print(f"Unique bus routes: {report['dataset_summary']['unique_routes']}")
    print(f"Unique vehicles: {report['dataset_summary']['unique_vehicles']}")
    
    # CUNY route findings
    if report['cuny_route_analysis']:
        print(f"\n=== CUNY ROUTE ANALYSIS ===")
        print("Top 5 CUNY-relevant routes by violation count:")
        for i, route in enumerate(report['cuny_route_analysis'][:5]):
            print(f"{i+1}. {route['bus_route_id']} ({route['borough']}): {route['violation_count']} violations")
    
    # Exempt vehicle findings
    if report['exempt_vehicle_analysis']:
        exempt_data = report['exempt_vehicle_analysis']
        print(f"\n=== EXEMPT VEHICLE ANALYSIS ===")
        print(f"Total exempt violations: {exempt_data.get('total_exempt_violations', 'N/A')}")
        print(f"Repeat offenders identified: {exempt_data.get('repeat_offender_count', 'N/A')}")
        
        if 'top_violation_locations' in exempt_data:
            print("Top violation locations for exempt vehicles:")
            for i, location in enumerate(exempt_data['top_violation_locations'][:3]):
                print(f"{i+1}. {location['location']}: {location['count']} violations ({location['exempt_type']})")
    
    # Congestion pricing findings
    if 'error' not in report['congestion_pricing_analysis']:
        cp_data = report['congestion_pricing_analysis']
        print(f"\n=== CONGESTION PRICING IMPACT ===")
        change_pct = cp_data['change_metrics']['violation_change_percent']
        print(f"Violation change since congestion pricing: {change_pct:.1f}%")
        print(f"Before CP daily average: {cp_data['before_period']['daily_average']:.1f}")
        print(f"After CP daily average: {cp_data['after_period']['daily_average']:.1f}")
    
    # Create visualizations
    print("\n=== CREATING VISUALIZATIONS ===")
    analyzer.create_visualizations(save_plots=True)
    
    # Save report
    with open('mta_ace_report.json', 'w') as f:
        # Convert DataFrame to dict for JSON serialization
        report_copy = report.copy()
        if isinstance(report_copy['cuny_route_analysis'], list):
            json.dump(report_copy, f, indent=2, default=str)
    print("Full report saved to 'mta_ace_report.json'")
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Use the analyzer object to perform additional custom analysis:")
    print("- analyzer.fetch_data_batch() for custom data queries")
    print("- analyzer.create_geographic_visualization() for interactive maps")
    print("- Access raw data through the API without local storage")


class CongestionPricingAnalyzer:
    """
    Specialized analyzer for congestion pricing impact on CBD bus routes
    """

    def __init__(self, ace_analyzer: MTAACEAnalyzer):
        self.ace_analyzer = ace_analyzer

        # Define CBD boundaries more precisely
        self.cbd_bounds = {
            'north': 60.0,  # 60th Street
            'south': 'Battery Park',  # Southern tip
            'east': 'East River',
            'west': 'Hudson River'
        }

        # CBD coordinates
        self.cbd_coordinates = {
            'north': 40.7829,  # ~60th Street
            'south': 40.7047,  # ~Battery Park
            'east': -73.9367,  # ~East River
            'west': -73.9903   # ~Hudson River
        }

        # Key CBD routes that are relevant for CUNY students
        self.cbd_routes = [
            'M15+', 'M15',  # First/Second Avenue
            'M2', 'M3',     # Fifth Avenue
            'M4', 'M5',     # Madison/6th Avenue
            'M42',          # 42nd Street Crosstown
            'M34+',         # 34th Street Select
            'M23+',         # 23rd Street Select
            'M14A', 'M14D', # 14th Street
            'M9', 'M11',    # Amsterdam/Columbus
            'M20', 'M21',   # 7th/8th Avenue
            'M60+',         # To LaGuardia (connects to many CUNY campuses)
            'M100', 'M101'  # Harlem routes serving CCNY area
        ]

        # Congestion pricing start date
        self.cp_start = datetime(2025, 1, 5)

        # Bus speed data API endpoints
        self.speed_apis = {
            'bus_speeds_2025': "https://data.ny.gov/resource/4u4b-jge6.json",  # MTA Bus Speeds 2025
            'route_segment_speeds_2025': "https://data.ny.gov/resource/kufs-yh3x.json"  # MTA Bus Route Segment Speeds 2025
        }

    def is_route_in_cbd(self, route_id: str) -> bool:
        """Check if a route operates within the CBD"""
        return route_id in self.cbd_routes

    def analyze_cbd_ace_violations(self, sample_size: int = 30000) -> pd.DataFrame:
        """
        Analyze ACE violations specifically for CBD routes
        """
        print("Analyzing ACE violations for CBD routes...")

        # Create filter for CBD routes
        cbd_filter = " OR ".join([f"bus_route_id = '{route}'" for route in self.cbd_routes])
        where_clause = f"({cbd_filter})"

        # Fetch data for CBD routes
        cbd_data = self.ace_analyzer.fetch_multiple_batches(
            total_records=sample_size,
            where_clause=where_clause,
            order_by="first_occurrence DESC"
        )

        if cbd_data.empty:
            print("No CBD ACE data found")
            return pd.DataFrame()

        # Add congestion pricing period classification
        cbd_data['cp_period'] = cbd_data['first_occurrence'].apply(
            lambda x: 'After' if x >= self.cp_start else 'Before'
        )

        return cbd_data

    def fetch_bus_speed_data(self, api_url: str, sample_size: int = 50000,
                           route_filter: str = None) -> pd.DataFrame:
        """
        Fetch bus speed data from MTA APIs with pagination
        """
        all_data = []
        batch_size = 1000
        total_batches = (sample_size + batch_size - 1) // batch_size

        print(f"Fetching bus speed data (up to {sample_size:,} records)...")

        for offset in range(0, sample_size, batch_size):
            current_limit = min(batch_size, sample_size - offset)

            params = {
                '$limit': current_limit,
                '$offset': offset
            }

            # Add route filter if specified - try different field names for different datasets
            if route_filter:
                params['$where'] = route_filter

            # Set appropriate order by field based on dataset
            if '4u4b-jge6' in api_url:  # Bus Speeds dataset
                params['$order'] = 'month DESC'
            elif 'kufs-yh3x' in api_url:  # Route Segment Speeds dataset
                params['$order'] = 'timestamp DESC'

            try:
                response = self.ace_analyzer.session.get(api_url, params=params)
                response.raise_for_status()
                data = response.json()

                if not data:
                    break

                batch_df = pd.DataFrame(data)
                if batch_df.empty:
                    break

                all_data.append(batch_df)

                # Show progress
                if (offset // batch_size + 1) % 10 == 0:
                    print(f"Fetched {len(all_data)} batches ({len(all_data) * batch_size:,} records)")

                # If we got fewer records than requested, we've reached the end
                if len(batch_df) < current_limit:
                    break

            except requests.RequestException as e:
                print(f"Error fetching speed data: {e}")
                break

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"Successfully fetched {len(combined_df):,} speed records")
            return combined_df
        else:
            print("No speed data fetched")
            return pd.DataFrame()

    def analyze_cbd_bus_speeds(self) -> Dict:
        """
        Analyze bus speeds for CBD routes from MTA datasets
        """
        print("Analyzing bus speeds for CBD routes...")

        # Create route filter for CBD routes
        cbd_routes_filter = " OR ".join([f"route_id = '{route}'" for route in self.cbd_routes])

        speed_analysis = {}

        # Fetch MTA Bus Speeds 2025
        print("1. Fetching MTA Bus Speeds 2025...")
        bus_speeds = self.fetch_bus_speed_data(
            self.speed_apis['bus_speeds_2025'],
            sample_size=30000,
            route_filter=cbd_routes_filter
        )

        # Fetch MTA Bus Route Segment Speeds 2025
        print("2. Fetching MTA Bus Route Segment Speeds 2025...")
        segment_speeds = self.fetch_bus_speed_data(
            self.speed_apis['route_segment_speeds_2025'],
            sample_size=50000,
            route_filter=cbd_routes_filter
        )

        # Analyze bus speeds
        if not bus_speeds.empty:
            speed_analysis['bus_speeds'] = self._analyze_speed_dataset(bus_speeds, 'Bus Speeds')

        # Analyze segment speeds
        if not segment_speeds.empty:
            speed_analysis['segment_speeds'] = self._analyze_speed_dataset(segment_speeds, 'Route Segment Speeds')

        # Combine analysis
        speed_analysis['combined_insights'] = self._generate_speed_insights(bus_speeds, segment_speeds)

        return speed_analysis

    def _analyze_speed_dataset(self, speed_df: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Analyze a speed dataset for CBD route patterns
        """
        analysis = {
            'dataset_name': dataset_name,
            'total_records': len(speed_df),
            'date_range': {},
            'route_analysis': {},
            'speed_statistics': {}
        }

        # Convert date columns if they exist
        date_columns = ['date', 'service_date', 'time_period']
        for col in date_columns:
            if col in speed_df.columns:
                try:
                    speed_df[col] = pd.to_datetime(speed_df[col])
                    analysis['date_range'] = {
                        'earliest': speed_df[col].min().strftime('%Y-%m-%d'),
                        'latest': speed_df[col].max().strftime('%Y-%m-%d')
                    }
                    break
                except:
                    continue

        # Analyze by route
        route_col = None
        for col in ['route_id', 'route', 'bus_route_id']:
            if col in speed_df.columns:
                route_col = col
                break

        if route_col:
            route_analysis = {}
            for route in self.cbd_routes:
                route_data = speed_df[speed_df[route_col] == route]
                if not route_data.empty:
                    route_stats = self._calculate_speed_stats(route_data)
                    route_analysis[route] = route_stats
            analysis['route_analysis'] = route_analysis

        # Overall speed statistics
        speed_cols = ['speed_mph', 'average_speed', 'speed', 'avg_speed_mph']
        for col in speed_cols:
            if col in speed_df.columns:
                try:
                    speeds = pd.to_numeric(speed_df[col], errors='coerce').dropna()
                    if not speeds.empty:
                        analysis['speed_statistics'] = {
                            'mean_speed': speeds.mean(),
                            'median_speed': speeds.median(),
                            'min_speed': speeds.min(),
                            'max_speed': speeds.max(),
                            'std_speed': speeds.std()
                        }
                        break
                except:
                    continue

        return analysis

    def _calculate_speed_stats(self, route_data: pd.DataFrame) -> Dict:
        """Calculate speed statistics for a specific route"""
        stats = {
            'record_count': len(route_data),
            'speed_stats': {}
        }

        # Find speed column
        speed_cols = ['speed_mph', 'average_speed', 'speed', 'avg_speed_mph']
        for col in speed_cols:
            if col in route_data.columns:
                try:
                    speeds = pd.to_numeric(route_data[col], errors='coerce').dropna()
                    if not speeds.empty:
                        stats['speed_stats'] = {
                            'mean': speeds.mean(),
                            'median': speeds.median(),
                            'min': speeds.min(),
                            'max': speeds.max()
                        }
                        break
                except:
                    continue

        # Analyze time periods if available
        if 'time_period' in route_data.columns:
            time_analysis = route_data['time_period'].value_counts().to_dict()
            stats['time_period_distribution'] = time_analysis

        return stats

    def _generate_speed_insights(self, bus_speeds: pd.DataFrame, segment_speeds: pd.DataFrame) -> Dict:
        """Generate insights combining speed datasets"""
        insights = {
            'total_speed_records': len(bus_speeds) + len(segment_speeds),
            'cbd_routes_with_data': [],
            'speed_comparison': {},
            'congestion_indicators': []
        }

        # Find routes with speed data
        for dataset, name in [(bus_speeds, 'bus_speeds'), (segment_speeds, 'segment_speeds')]:
            if not dataset.empty:
                route_cols = ['route_id', 'route', 'bus_route_id']
                for col in route_cols:
                    if col in dataset.columns:
                        routes_with_data = dataset[col].unique()
                        cbd_routes_with_data = [r for r in routes_with_data if r in self.cbd_routes]
                        insights['cbd_routes_with_data'].extend(cbd_routes_with_data)
                        break

        insights['cbd_routes_with_data'] = list(set(insights['cbd_routes_with_data']))

        # Generate congestion indicators
        if insights['cbd_routes_with_data']:
            insights['congestion_indicators'] = [
                f"Speed data available for {len(insights['cbd_routes_with_data'])} CBD routes",
                "Analysis covers routes serving major CUNY campuses",
                "Data can show congestion pricing impact on bus performance"
            ]

        return insights

    def create_speed_comparison_map(self, speed_analysis: Dict, cbd_violations: pd.DataFrame) -> folium.Map:
        """
        Create a map showing bus speeds and violations for CBD routes
        """
        # Create base map centered on CBD
        center_lat = (self.cbd_coordinates['north'] + self.cbd_coordinates['south']) / 2
        center_lon = (self.cbd_coordinates['east'] + self.cbd_coordinates['west']) / 2

        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

        # Add CBD boundary
        cbd_polygon = [
            [self.cbd_coordinates['north'], self.cbd_coordinates['west']],
            [self.cbd_coordinates['north'], self.cbd_coordinates['east']],
            [self.cbd_coordinates['south'], self.cbd_coordinates['east']],
            [self.cbd_coordinates['south'], self.cbd_coordinates['west']]
        ]

        folium.Polygon(
            locations=cbd_polygon,
            color='red',
            weight=3,
            fillColor='red',
            fillOpacity=0.1,
            popup='Congestion Pricing Zone (CBD)'
        ).add_to(m)

        # Add route-specific speed information as markers
        if 'bus_speeds' in speed_analysis and 'route_analysis' in speed_analysis['bus_speeds']:
            route_speeds = speed_analysis['bus_speeds']['route_analysis']

            # Create markers for routes with speed data
            route_positions = {
                'M15+': [40.7505, -73.9934],  # First Avenue
                'M2': [40.7614, -73.9776],    # Fifth Avenue
                'M42': [40.7589, -73.9851],   # 42nd Street
                'M101': [40.7831, -73.9712],  # Lexington Avenue
                'M100': [40.7831, -73.9544],  # Amsterdam Avenue
                'M34+': [40.7505, -73.9897],  # 34th Street
                'M60+': [40.7505, -73.9897]   # LaGuardia connection
            }

            for route, position in route_positions.items():
                if route in route_speeds and 'speed_stats' in route_speeds[route]:
                    speed_stats = route_speeds[route]['speed_stats']
                    if speed_stats:
                        avg_speed = speed_stats.get('mean', 0)

                        # Color code by speed (green = faster, red = slower)
                        if avg_speed > 12:
                            color = 'green'
                        elif avg_speed > 8:
                            color = 'orange'
                        else:
                            color = 'red'

                        folium.Marker(
                            location=position,
                            popup=f"Route {route}<br>Avg Speed: {avg_speed:.1f} mph<br>Records: {route_speeds[route]['record_count']}",
                            icon=folium.Icon(color=color, icon='bus', prefix='fa')
                        ).add_to(m)

        # Add legend
        legend_html = '''
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 250px; height: 140px;
                    background-color: white; border:2px solid grey; z-index:9999;
                    font-size:14px; padding: 10px">
        <p><b>CBD Bus Speed Analysis</b></p>
        <p><i class="fa fa-bus" style="color:green"></i> Fast Routes (>12 mph)</p>
        <p><i class="fa fa-bus" style="color:orange"></i> Moderate (8-12 mph)</p>
        <p><i class="fa fa-bus" style="color:red"></i> Slow Routes (<8 mph)</p>
        <p><i class="fa fa-square" style="color:red; opacity:0.3"></i> CBD Zone</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        return m

    def compare_before_after_violations(self, cbd_data: pd.DataFrame) -> Dict:
        """
        Compare violations before and after congestion pricing
        """
        if cbd_data.empty:
            return {}

        before_data = cbd_data[cbd_data['cp_period'] == 'Before']
        after_data = cbd_data[cbd_data['cp_period'] == 'After']

        analysis = {
            'before_cp': {
                'total_violations': len(before_data),
                'daily_average': len(before_data) / max(1, (self.cp_start - before_data['first_occurrence'].min()).days) if len(before_data) > 0 else 0,
                'route_breakdown': before_data['bus_route_id'].value_counts().to_dict(),
                'violation_types': before_data['violation_type'].value_counts().to_dict(),
                'date_range': {
                    'start': before_data['first_occurrence'].min().strftime('%Y-%m-%d') if len(before_data) > 0 else 'No data',
                    'end': before_data['first_occurrence'].max().strftime('%Y-%m-%d') if len(before_data) > 0 else 'No data'
                }
            },
            'after_cp': {
                'total_violations': len(after_data),
                'daily_average': len(after_data) / max(1, (after_data['first_occurrence'].max() - self.cp_start).days) if len(after_data) > 0 else 0,
                'route_breakdown': after_data['bus_route_id'].value_counts().to_dict(),
                'violation_types': after_data['violation_type'].value_counts().to_dict(),
                'date_range': {
                    'start': after_data['first_occurrence'].min().strftime('%Y-%m-%d') if len(after_data) > 0 else 'No data',
                    'end': after_data['first_occurrence'].max().strftime('%Y-%m-%d') if len(after_data) > 0 else 'No data'
                }
            }
        }

        # Calculate changes
        if analysis['before_cp']['total_violations'] > 0:
            violation_change = ((analysis['after_cp']['total_violations'] - analysis['before_cp']['total_violations'])
                              / analysis['before_cp']['total_violations']) * 100
            daily_change = analysis['after_cp']['daily_average'] - analysis['before_cp']['daily_average']
        else:
            violation_change = 0
            daily_change = 0

        analysis['changes'] = {
            'violation_change_percent': violation_change,
            'daily_average_change': daily_change,
            'interpretation': self._interpret_changes(violation_change, daily_change)
        }

        return analysis

    def _interpret_changes(self, violation_change: float, daily_change: float) -> str:
        """Interpret the congestion pricing impact"""
        if violation_change < -10:
            return "Significant decrease in violations - congestion pricing may be improving traffic flow"
        elif violation_change > 10:
            return "Significant increase in violations - may indicate increased enforcement or traffic issues"
        else:
            return "Minimal change in violations - congestion pricing impact unclear from ACE data alone"

    def create_cbd_violation_map(self, cbd_data: pd.DataFrame) -> folium.Map:
        """
        Create a map showing CBD violations before and after congestion pricing
        """
        if cbd_data.empty:
            return None

        # Filter for valid coordinates within CBD
        map_data = cbd_data.dropna(subset=['violation_latitude', 'violation_longitude'])
        map_data = map_data[
            (map_data['violation_latitude'] >= self.cbd_coordinates['south']) &
            (map_data['violation_latitude'] <= self.cbd_coordinates['north']) &
            (map_data['violation_longitude'] >= self.cbd_coordinates['west']) &
            (map_data['violation_longitude'] <= self.cbd_coordinates['east'])
        ]

        if map_data.empty:
            print("No valid CBD coordinate data for mapping")
            return None

        # Create base map centered on CBD
        center_lat = (self.cbd_coordinates['north'] + self.cbd_coordinates['south']) / 2
        center_lon = (self.cbd_coordinates['east'] + self.cbd_coordinates['west']) / 2

        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

        # Add CBD boundary
        cbd_polygon = [
            [self.cbd_coordinates['north'], self.cbd_coordinates['west']],
            [self.cbd_coordinates['north'], self.cbd_coordinates['east']],
            [self.cbd_coordinates['south'], self.cbd_coordinates['east']],
            [self.cbd_coordinates['south'], self.cbd_coordinates['west']]
        ]

        folium.Polygon(
            locations=cbd_polygon,
            color='red',
            weight=3,
            fillColor='red',
            fillOpacity=0.1,
            popup='Congestion Pricing Zone (CBD)'
        ).add_to(m)

        # Color code by time period
        period_colors = {
            'Before': 'blue',
            'After': 'orange'
        }

        # Add violation points
        for _, row in map_data.head(1000).iterrows():  # Limit for performance
            color = period_colors.get(row['cp_period'], 'gray')

            folium.CircleMarker(
                location=[row['violation_latitude'], row['violation_longitude']],
                radius=4,
                popup=f"Route: {row['bus_route_id']}<br>Period: {row['cp_period']}<br>Type: {row['violation_type']}<br>Date: {row['first_occurrence'].strftime('%Y-%m-%d')}",
                color=color,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)

        # Add legend
        legend_html = '''
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 220px; height: 120px;
                    background-color: white; border:2px solid grey; z-index:9999;
                    font-size:14px; padding: 10px">
        <p><b>Congestion Pricing Impact</b></p>
        <p><i class="fa fa-circle" style="color:blue"></i> Before CP (Jan 5, 2025)</p>
        <p><i class="fa fa-circle" style="color:orange"></i> After CP (Jan 5, 2025)</p>
        <p><i class="fa fa-square" style="color:red; opacity:0.3"></i> CBD Zone</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        return m

    def generate_cbd_impact_report(self) -> Dict:
        """
        Generate comprehensive CBD congestion pricing impact report
        """
        print("=== CBD CONGESTION PRICING IMPACT ANALYSIS ===")

        # Analyze ACE violations
        print("1. Analyzing ACE violations for CBD routes...")
        cbd_violations = self.analyze_cbd_ace_violations(sample_size=50000)

        # Compare before/after
        print("2. Comparing before/after congestion pricing...")
        violation_analysis = self.compare_before_after_violations(cbd_violations)

        # Analyze bus speeds
        print("3. Analyzing bus speeds for CBD routes...")
        speed_analysis = self.analyze_cbd_bus_speeds()

        # Create visualizations
        print("4. Creating comprehensive visualizations...")
        cbd_map = self.create_cbd_violation_map(cbd_violations)
        speed_map = self.create_speed_comparison_map(speed_analysis, cbd_violations)

        # Route-specific analysis
        print("5. Analyzing route-specific impacts...")
        route_analysis = self._analyze_route_specific_impacts(cbd_violations)

        # Generate integrated insights
        print("6. Generating integrated insights...")
        integrated_insights = self._generate_integrated_insights(violation_analysis, speed_analysis)

        report = {
            'analysis_date': datetime.now().isoformat(),
            'cbd_routes_analyzed': self.cbd_routes,
            'congestion_pricing_start': self.cp_start.isoformat(),
            'violation_analysis': violation_analysis,
            'speed_analysis': speed_analysis,
            'route_specific_analysis': route_analysis,
            'integrated_insights': integrated_insights,
            'total_records_analyzed': len(cbd_violations),
            'violations_map_created': cbd_map is not None,
            'speed_map_created': speed_map is not None,
            'recommendations': self._generate_comprehensive_recommendations(violation_analysis, speed_analysis)
        }

        return report

    def _analyze_route_specific_impacts(self, cbd_data: pd.DataFrame) -> Dict:
        """Analyze impact on specific CBD routes"""
        if cbd_data.empty:
            return {}

        route_impacts = {}

        for route in self.cbd_routes:
            route_data = cbd_data[cbd_data['bus_route_id'] == route]
            if route_data.empty:
                continue

            before = route_data[route_data['cp_period'] == 'Before']
            after = route_data[route_data['cp_period'] == 'After']

            if len(before) > 0 and len(after) > 0:
                change = ((len(after) - len(before)) / len(before)) * 100
                route_impacts[route] = {
                    'before_violations': len(before),
                    'after_violations': len(after),
                    'change_percent': change,
                    'status': 'Improved' if change < -5 else 'Worsened' if change > 5 else 'Stable'
                }

        return route_impacts

    def _generate_integrated_insights(self, violation_analysis: Dict, speed_analysis: Dict) -> Dict:
        """Generate insights combining violation and speed data"""
        insights = {
            'data_coverage': {},
            'congestion_indicators': [],
            'cuny_impact_assessment': [],
            'policy_implications': []
        }

        # Data coverage analysis
        insights['data_coverage'] = {
            'violation_records': violation_analysis.get('after_cp', {}).get('total_violations', 0),
            'speed_records': speed_analysis.get('combined_insights', {}).get('total_speed_records', 0),
            'routes_with_speed_data': len(speed_analysis.get('combined_insights', {}).get('cbd_routes_with_data', [])),
            'total_cbd_routes_analyzed': len(self.cbd_routes)
        }

        # Congestion indicators
        if speed_analysis.get('bus_speeds', {}).get('speed_statistics'):
            avg_speed = speed_analysis['bus_speeds']['speed_statistics'].get('mean_speed', 0)
            if avg_speed > 0:
                if avg_speed < 8:
                    insights['congestion_indicators'].append("Low average speeds indicate significant congestion in CBD")
                elif avg_speed > 12:
                    insights['congestion_indicators'].append("Higher average speeds suggest improved traffic flow")
                else:
                    insights['congestion_indicators'].append("Moderate speeds indicate typical urban traffic conditions")

        # CUNY impact assessment
        cuny_critical_routes = ['M15+', 'M2', 'M101', 'M100', 'M42', 'M60+']
        routes_with_data = speed_analysis.get('combined_insights', {}).get('cbd_routes_with_data', [])
        cuny_routes_covered = [r for r in cuny_critical_routes if r in routes_with_data]

        insights['cuny_impact_assessment'] = [
            f"Analysis covers {len(cuny_routes_covered)} out of {len(cuny_critical_routes)} critical CUNY routes",
            "Routes analyzed serve major campuses: CCNY, Hunter, Baruch, and connections",
            "Speed data can inform student commute planning and campus accessibility"
        ]

        # Policy implications
        violation_change = violation_analysis.get('changes', {}).get('violation_change_percent', 0)
        if abs(violation_change) < 5:
            insights['policy_implications'].append("Minimal violation changes suggest congestion pricing has stabilized traffic patterns")
        elif violation_change > 10:
            insights['policy_implications'].append("Increased violations may indicate traffic displacement to bus routes")
        else:
            insights['policy_implications'].append("Decreased violations suggest improved traffic compliance in CBD")

        insights['policy_implications'].extend([
            "Speed data provides objective measure of congestion pricing effectiveness",
            "Bus performance metrics essential for evaluating public transit benefits",
            "Continued monitoring needed to assess long-term impacts on CUNY student mobility"
        ])

        return insights

    def _generate_comprehensive_recommendations(self, violation_analysis: Dict, speed_analysis: Dict) -> List[str]:
        """Generate comprehensive recommendations based on both violation and speed analysis"""
        recommendations = []

        # Speed-based recommendations
        if speed_analysis.get('combined_insights', {}).get('total_speed_records', 0) > 0:
            recommendations.extend([
                "Continue monitoring bus speeds as primary indicator of congestion pricing success",
                "Use speed data to optimize bus service frequency during peak hours",
                "Correlate speed improvements with violation reductions for policy assessment"
            ])

            # Route-specific recommendations
            if speed_analysis.get('bus_speeds', {}).get('route_analysis'):
                recommendations.append("Prioritize speed improvements on routes with consistently low performance")
        else:
            recommendations.append("Expand speed data collection to better assess congestion pricing impact")

        # Violation-based recommendations
        violation_change = violation_analysis.get('changes', {}).get('violation_change_percent', 0)
        if violation_change > 10:
            recommendations.extend([
                "Investigate causes of increased ACE violations in CBD",
                "Review if congestion pricing has shifted traffic patterns affecting bus lanes",
                "Consider enhanced enforcement during peak congestion pricing hours"
            ])
        elif violation_change < -10:
            recommendations.extend([
                "Document successful violation reduction strategies for replication",
                "Maintain current enforcement levels to sustain improvements"
            ])

        # CUNY-specific recommendations
        recommendations.extend([
            "Monitor student commute patterns on key routes (M15+, M101, M2, M42)",
            "Coordinate with CUNY campuses to assess transportation accessibility changes",
            "Consider transit subsidies if congestion pricing negatively impacts student mobility",
            "Use data to inform campus shuttle service adjustments"
        ])

        # Data integration recommendations
        recommendations.extend([
            "Establish regular reporting combining ACE violations and speed data",
            "Create dashboard for real-time monitoring of CBD bus performance",
            "Integrate with MTA ridership data for comprehensive transit analysis"
        ])

        return recommendations

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        if not analysis:
            return ["Insufficient data for recommendations"]

        change_pct = analysis.get('changes', {}).get('violation_change_percent', 0)

        if change_pct < -10:
            recommendations.extend([
                "Congestion pricing appears to be reducing ACE violations",
                "Continue monitoring to confirm trend sustainability",
                "Consider expanding similar pricing to other high-traffic areas"
            ])
        elif change_pct > 10:
            recommendations.extend([
                "Investigate causes of increased violations post-congestion pricing",
                "Review if pricing has shifted traffic patterns to bus routes",
                "Consider adjusting enforcement strategies for changed traffic patterns"
            ])
        else:
            recommendations.extend([
                "Violation patterns show minimal change from congestion pricing",
                "Supplement ACE data with bus speed and ridership data for fuller picture",
                "Consider longer observation period to identify trends"
            ])

        recommendations.append("Integrate MTA Bus Speed data for comprehensive analysis")
        recommendations.append("Monitor CUNY student commute patterns on affected routes")

        return recommendations


def analyze_congestion_pricing_impact():
    """
    Main function to analyze congestion pricing impact on CBD bus routes
    """
    print("=== CONGESTION PRICING IMPACT ON CBD BUS ROUTES ===")

    # Initialize analyzers
    ace_analyzer = MTAACEAnalyzer()
    cp_analyzer = CongestionPricingAnalyzer(ace_analyzer)

    # Generate comprehensive report
    report = cp_analyzer.generate_cbd_impact_report()

    # Print key findings
    print("\n=== KEY FINDINGS ===")
    if report['violation_analysis']:
        va = report['violation_analysis']
        print(f"Before CP violations: {va['before_cp']['total_violations']:,}")
        print(f"After CP violations: {va['after_cp']['total_violations']:,}")
        print(f"Change: {va['changes']['violation_change_percent']:+.1f}%")
        print(f"Daily average change: {va['changes']['daily_average_change']:+.1f}")
        print(f"Interpretation: {va['changes']['interpretation']}")

    # Route-specific impacts
    if report['route_specific_analysis']:
        print(f"\n=== ROUTE-SPECIFIC IMPACTS ===")
        for route, impact in list(report['route_specific_analysis'].items())[:5]:
            print(f"{route}: {impact['change_percent']:+.1f}% ({impact['status']})")

    # Save comprehensive report
    with open('cbd_congestion_pricing_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to 'cbd_congestion_pricing_report.json'")

    # Save maps if created
    if report['violations_map_created']:
        violations_map = cp_analyzer.create_cbd_violation_map(
            cp_analyzer.analyze_cbd_ace_violations(sample_size=10000)
        )
        if violations_map:
            violations_map.save('cbd_violations_map.html')
            print("Violations map saved to 'cbd_violations_map.html'")

    if report['speed_map_created']:
        speed_map = cp_analyzer.create_speed_comparison_map(
            report['speed_analysis'],
            cp_analyzer.analyze_cbd_ace_violations(sample_size=5000)
        )
        if speed_map:
            speed_map.save('cbd_speed_analysis_map.html')
            print("Speed analysis map saved to 'cbd_speed_analysis_map.html'")

    return report


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--cbd-analysis':
        analyze_congestion_pricing_impact()
    else:
        main()