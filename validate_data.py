#!/usr/bin/env python3
"""
Data Validation Utility for Scientific Hypertrophy Trainer

Comprehensive validation tool for training data CSV files.
Checks format, data quality, and provides recommendations.

Usage:
    python validate_data.py data.csv
    python validate_data.py data.csv --detailed
    python validate_data.py data.csv --fix-output cleaned_data.csv

Author: Scientific Hypertrophy Trainer Team
"""

import pandas as pd
import numpy as np
import sys
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataValidator:
    """Comprehensive training data validator."""
    
    def __init__(self):
        # Required columns (minimum for basic functionality)
        self.required_columns = ['total_sets']
        
        # Recommended columns for best predictions
        self.recommended_columns = [
            'date', 'average_rpe', 'sleep_quality_1_10', 
            'sleep_duration_hours', 'hrv_rmssd', 'perceived_stress_1_10'
        ]
        
        # Optional but useful columns
        self.useful_columns = [
            'user_name', 'calories', 'protein_g', 'weight_kg',
            'session_duration_min', 'compound_sets', 'isolation_sets'
        ]
        
        # Valid data ranges
        self.data_ranges = {
            'total_sets': (0, 50),
            'average_rpe': (1, 10),
            'sleep_quality_1_10': (1, 10),
            'sleep_duration_hours': (3, 12),
            'hrv_rmssd': (10, 100),
            'perceived_stress_1_10': (1, 10),
            'calories': (800, 5000),
            'protein_g': (50, 400),
            'weight_kg': (40, 200)
        }
    
    def validate_file(self, file_path: str, detailed: bool = False) -> Dict:
        """
        Validate training data file comprehensively.
        
        Args:
            file_path: Path to CSV file
            detailed: Whether to include detailed analysis
            
        Returns:
            Validation results dictionary
        """
        results = {
            'file_path': file_path,
            'timestamp': datetime.now().isoformat(),
            'valid': False,
            'errors': [],
            'warnings': [],
            'info': {},
            'recommendations': []
        }
        
        try:
            # Load data
            df = pd.read_csv(file_path)
            results['info']['total_rows'] = len(df)
            results['info']['total_columns'] = len(df.columns)
            results['info']['columns'] = list(df.columns)
            
            if len(df) == 0:
                results['errors'].append("File is empty - no data to validate")
                return results
            
            # Run validation checks
            self._check_required_columns(df, results)
            self._check_column_quality(df, results)
            self._check_data_ranges(df, results)
            self._check_data_consistency(df, results)
            self._check_temporal_data(df, results)
            
            if detailed:
                self._detailed_analysis(df, results)
            
            # Generate recommendations
            self._generate_recommendations(df, results)
            
            # Final validation status
            results['valid'] = len(results['errors']) == 0
            
        except FileNotFoundError:
            results['errors'].append(f"File not found: {file_path}")
        except pd.errors.EmptyDataError:
            results['errors'].append("File is empty or contains no data")
        except pd.errors.ParserError as e:
            results['errors'].append(f"CSV parsing error: {str(e)}")
        except Exception as e:
            results['errors'].append(f"Unexpected error: {str(e)}")
        
        return results
    
    def _check_required_columns(self, df: pd.DataFrame, results: Dict):
        """Check for required columns."""
        missing_required = [col for col in self.required_columns if col not in df.columns]
        
        if missing_required:
            results['errors'].append(f"Missing required columns: {missing_required}")
        
        missing_recommended = [col for col in self.recommended_columns if col not in df.columns]
        if missing_recommended:
            results['warnings'].append(f"Missing recommended columns: {missing_recommended}")
            results['recommendations'].append(
                f"Add columns {missing_recommended} for better predictions"
            )
    
    def _check_column_quality(self, df: pd.DataFrame, results: Dict):
        """Check quality of individual columns."""
        
        for col in df.columns:
            if col in self.required_columns or col in self.recommended_columns:
                
                # Check for completely null columns
                null_percentage = df[col].isnull().sum() / len(df) * 100
                
                if null_percentage == 100:
                    results['errors'].append(f"Column '{col}' is completely empty")
                elif null_percentage > 50:
                    results['warnings'].append(f"Column '{col}' is {null_percentage:.1f}% empty")
                elif null_percentage > 20:
                    results['warnings'].append(f"Column '{col}' has {null_percentage:.1f}% missing values")
                
                # Check for constant values
                if df[col].nunique() == 1 and not df[col].isnull().all():
                    results['warnings'].append(f"Column '{col}' has constant value: {df[col].iloc[0]}")
    
    def _check_data_ranges(self, df: pd.DataFrame, results: Dict):
        """Check if data values are within reasonable ranges."""
        
        for col, (min_val, max_val) in self.data_ranges.items():
            if col in df.columns and not df[col].isnull().all():
                
                col_min = df[col].min()
                col_max = df[col].max()
                
                # Check for values outside expected range
                out_of_range_low = (df[col] < min_val).sum()
                out_of_range_high = (df[col] > max_val).sum()
                
                if out_of_range_low > 0:
                    results['warnings'].append(
                        f"Column '{col}': {out_of_range_low} values below expected range (< {min_val})"
                    )
                
                if out_of_range_high > 0:
                    results['warnings'].append(
                        f"Column '{col}': {out_of_range_high} values above expected range (> {max_val})"
                    )
                
                # Store actual ranges for info
                results['info'][f'{col}_range'] = (float(col_min), float(col_max))
    
    def _check_data_consistency(self, df: pd.DataFrame, results: Dict):
        """Check for logical consistency in the data."""
        
        # Check for negative values where they shouldn't exist
        non_negative_cols = ['total_sets', 'calories', 'protein_g', 'weight_kg', 
                            'sleep_duration_hours', 'session_duration_min']
        
        for col in non_negative_cols:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    results['errors'].append(f"Column '{col}' has {negative_count} negative values")
        
        # Check for duplicate dates (if user and date columns exist)
        if 'date' in df.columns and 'user_name' in df.columns:
            df_temp = df.copy()
            df_temp['date'] = pd.to_datetime(df_temp['date'], errors='coerce')
            
            duplicates = df_temp.groupby(['user_name', 'date']).size()
            duplicate_count = (duplicates > 1).sum()
            
            if duplicate_count > 0:
                results['warnings'].append(f"Found {duplicate_count} duplicate date-user combinations")
        
        # Check training volume consistency
        if 'total_sets' in df.columns and 'compound_sets' in df.columns and 'isolation_sets' in df.columns:
            inconsistent = df['total_sets'] != (df['compound_sets'] + df['isolation_sets'])
            inconsistent_count = inconsistent.sum()
            
            if inconsistent_count > 0:
                results['warnings'].append(
                    f"Training volume inconsistency: {inconsistent_count} rows where "
                    "total_sets ≠ compound_sets + isolation_sets"
                )
    
    def _check_temporal_data(self, df: pd.DataFrame, results: Dict):
        """Check temporal aspects of the data."""
        
        if 'date' in df.columns:
            try:
                df_temp = df.copy()
                df_temp['date'] = pd.to_datetime(df_temp['date'], errors='coerce')
                
                # Check for invalid dates
                invalid_dates = df_temp['date'].isnull().sum()
                if invalid_dates > 0:
                    results['warnings'].append(f"Found {invalid_dates} invalid/unparseable dates")
                
                valid_dates = df_temp['date'].dropna()
                if len(valid_dates) > 0:
                    date_range = valid_dates.max() - valid_dates.min()
                    results['info']['date_span_days'] = date_range.days
                    results['info']['earliest_date'] = valid_dates.min().strftime('%Y-%m-%d')
                    results['info']['latest_date'] = valid_dates.max().strftime('%Y-%m-%d')
                    
                    # Check if data is recent
                    days_old = (datetime.now() - valid_dates.max()).days
                    if days_old > 14:
                        results['warnings'].append(f"Data is {days_old} days old - recent data improves predictions")
                    
                    # Check for reasonable date range
                    if date_range.days > 730:  # 2 years
                        results['info']['note'] = "Very long date range - consider analyzing recent periods separately"
                    elif date_range.days < 7:
                        results['warnings'].append("Date range less than 1 week - may limit prediction accuracy")
                
            except Exception as e:
                results['warnings'].append(f"Error analyzing dates: {str(e)}")
    
    def _detailed_analysis(self, df: pd.DataFrame, results: Dict):
        """Perform detailed statistical analysis."""
        
        results['detailed_stats'] = {}
        
        # Analyze numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in self.data_ranges or col in self.required_columns + self.recommended_columns:
                stats = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'null_count': int(df[col].isnull().sum()),
                    'unique_values': int(df[col].nunique())
                }
                results['detailed_stats'][col] = stats
        
        # Training frequency analysis
        if 'total_sets' in df.columns:
            training_days = (df['total_sets'] > 0).sum()
            total_days = len(df)
            training_frequency = training_days / total_days * 100
            
            results['info']['training_frequency'] = f"{training_frequency:.1f}%"
            
            if training_frequency < 30:
                results['warnings'].append(f"Low training frequency ({training_frequency:.1f}%)")
            elif training_frequency > 80:
                results['warnings'].append(f"Very high training frequency ({training_frequency:.1f}%) - ensure adequate recovery")
        
        # User analysis
        if 'user_name' in df.columns:
            user_counts = df['user_name'].value_counts()
            results['info']['users'] = user_counts.to_dict()
            results['info']['user_count'] = len(user_counts)
            
            # Check for users with insufficient data
            insufficient_users = user_counts[user_counts < 7]
            if len(insufficient_users) > 0:
                results['warnings'].append(
                    f"Users with <7 days of data: {list(insufficient_users.index)}"
                )
    
    def _generate_recommendations(self, df: pd.DataFrame, results: Dict):
        """Generate actionable recommendations."""
        
        # Data collection recommendations
        if 'date' not in df.columns:
            results['recommendations'].append("Add 'date' column to track temporal patterns")
        
        if 'user_name' not in df.columns and len(df) > 30:
            results['recommendations'].append("Add 'user_name' column if tracking multiple people")
        
        # Missing critical features
        critical_missing = [col for col in ['average_rpe', 'sleep_quality_1_10'] 
                           if col not in df.columns]
        if critical_missing:
            results['recommendations'].append(
                f"Add {critical_missing} for significantly better predictions"
            )
        
        # Data quality improvements
        if 'total_sets' in df.columns:
            zero_days = (df['total_sets'] == 0).sum()
            if zero_days > len(df) * 0.7:
                results['recommendations'].append("Consider tracking rest days separately from training days")
        
        # Prediction readiness
        if len(results['errors']) == 0:
            if len(df) >= 14:
                results['recommendations'].append("✅ Data ready for reliable predictions")
            elif len(df) >= 7:
                results['recommendations'].append("Data sufficient for basic predictions - collect more for better accuracy")
            else:
                results['recommendations'].append("Collect at least 7 days of data for meaningful predictions")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='🔍 Data Validation Tool for Scientific Hypertrophy Trainer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  Basic validation:
    python validate_data.py training_data.csv
  
  Detailed analysis:
    python validate_data.py training_data.csv --detailed
  
  Generate cleaned data:
    python validate_data.py training_data.csv --fix-output cleaned_data.csv
  
  JSON output for automation:
    python validate_data.py training_data.csv --json
        """
    )
    
    parser.add_argument('file_path', help='Path to training data CSV file')
    parser.add_argument('--detailed', action='store_true', 
                       help='Include detailed statistical analysis')
    parser.add_argument('--json', action='store_true',
                       help='Output results as JSON')
    parser.add_argument('--fix-output', metavar='FILE',
                       help='Generate cleaned data file with fixes applied')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🔍 Scientific Hypertrophy Trainer - Data Validator")
        print("=" * 60)
        print(f"📂 Analyzing: {args.file_path}")
    
    # Validate the file
    validator = DataValidator()
    results = validator.validate_file(args.file_path, detailed=args.detailed)
    
    # Output results
    if args.json:
        import json
        print(json.dumps(results, indent=2))
    else:
        display_results(results, quiet=args.quiet)
    
    # Generate fixed file if requested
    if args.fix_output:
        try:
            generate_cleaned_data(args.file_path, args.fix_output, results)
        except Exception as e:
            print(f"❌ Failed to generate cleaned data: {e}")
    
    # Exit with appropriate code
    sys.exit(0 if results['valid'] else 1)

def display_results(results: Dict, quiet: bool = False):
    """Display validation results in human-readable format."""
    
    # Summary
    status_icon = "✅" if results['valid'] else "❌"
    status_text = "VALID" if results['valid'] else "INVALID"
    
    print(f"\n{status_icon} VALIDATION STATUS: {status_text}")
    
    # Basic info
    if not quiet:
        info = results['info']
        print(f"📊 Dataset Info:")
        print(f"   Rows: {info['total_rows']:,}")
        print(f"   Columns: {info['total_columns']}")
        
        if 'date_span_days' in info:
            print(f"   Date Range: {info['earliest_date']} to {info['latest_date']} ({info['date_span_days']} days)")
        
        if 'user_count' in info:
            print(f"   Users: {info['user_count']}")
        
        if 'training_frequency' in info:
            print(f"   Training Frequency: {info['training_frequency']}")
    
    # Errors
    if results['errors']:
        print(f"\n❌ ERRORS ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"   • {error}")
    
    # Warnings  
    if results['warnings']:
        print(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
        for warning in results['warnings']:
            print(f"   • {warning}")
    
    # Recommendations
    if results['recommendations']:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in results['recommendations']:
            print(f"   • {rec}")
    
    # Detailed stats
    if 'detailed_stats' in results and not quiet:
        print(f"\n📈 DETAILED STATISTICS:")
        for col, stats in results['detailed_stats'].items():
            print(f"   {col}:")
            print(f"     Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
            print(f"     Range: {stats['min']:.2f} - {stats['max']:.2f}")
            if stats['null_count'] > 0:
                print(f"     Missing: {stats['null_count']}")

def generate_cleaned_data(input_file: str, output_file: str, results: Dict):
    """Generate cleaned version of the data."""
    
    print(f"\n🧹 Generating cleaned data: {output_file}")
    
    df = pd.read_csv(input_file)
    cleaned = df.copy()
    
    # Fix negative values in non-negative columns
    non_negative_cols = ['total_sets', 'calories', 'protein_g', 'weight_kg', 
                        'sleep_duration_hours', 'session_duration_min']
    
    for col in non_negative_cols:
        if col in cleaned.columns:
            negative_mask = cleaned[col] < 0
            if negative_mask.any():
                cleaned.loc[negative_mask, col] = 0
                print(f"   Fixed {negative_mask.sum()} negative values in '{col}'")
    
    # Clip values to reasonable ranges
    validator = DataValidator()
    for col, (min_val, max_val) in validator.data_ranges.items():
        if col in cleaned.columns:
            original_out = ((cleaned[col] < min_val) | (cleaned[col] > max_val)).sum()
            cleaned[col] = np.clip(cleaned[col], min_val, max_val)
            if original_out > 0:
                print(f"   Clipped {original_out} out-of-range values in '{col}'")
    
    # Remove duplicate date-user combinations (keep last)
    if 'date' in cleaned.columns and 'user_name' in cleaned.columns:
        before_len = len(cleaned)
        cleaned = cleaned.drop_duplicates(subset=['date', 'user_name'], keep='last')
        removed = before_len - len(cleaned)
        if removed > 0:
            print(f"   Removed {removed} duplicate date-user combinations")
    
    # Save cleaned data
    cleaned.to_csv(output_file, index=False)
    print(f"✅ Cleaned data saved: {output_file}")
    print(f"   Original: {len(df)} rows")
    print(f"   Cleaned: {len(cleaned)} rows")

if __name__ == '__main__':
    main()
