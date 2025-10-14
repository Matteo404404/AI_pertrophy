#!/usr/bin/env python3
"""
Scientific Hypertrophy Trainer - Command Line Interface

Production-ready CLI for making evidence-based training predictions.

Usage:
    python trainer.py --demo                    # Demo with sample data
    python trainer.py data.csv                  # Analyze your data
    python trainer.py data.csv --user Alex      # Analyze specific user
    python trainer.py --generate-data 90        # Generate synthetic data

Author: Scientific Hypertrophy Trainer Team
"""

import argparse
import pandas as pd
import sys
import os
import json
from datetime import datetime
from typing import Optional

# Add current directory to path for imports
sys.path.insert(0, os.getcwd())

# ASCII Art Header
HEADER = """
🏋️  SCIENTIFIC HYPERTROPHY TRAINER v1.0
═══════════════════════════════════════════════════════════
Evidence-based training predictions • RIR recommendations
Rest time optimization • Recovery analysis
"""

def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    print(HEADER)
    
    try:
        if args.demo:
            run_demo()
        elif args.generate_data:
            generate_data(args.generate_data, args.output)
        elif args.validate and args.data_file:
            validate_data_file(args.data_file)
        elif args.data_file:
            analyze_data(args.data_file, args.user, args.format, args.days, args.output)
        else:
            print("❌ No action specified. Use --help for usage information.")
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n\n👋 Training session interrupted. Stay strong!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.debug if hasattr(args, 'debug') else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='🏋️ Scientific Hypertrophy Trainer - Evidence-based training predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  Demo with sample data:
    python trainer.py --demo
  
  Analyze your training data:
    python trainer.py my_training_log.csv
  
  Analyze specific user:
    python trainer.py data.csv --user "Alex_Beginner"
  
  Generate synthetic training data:
    python trainer.py --generate-data 90 --output training_data.csv
  
  Get JSON output for API integration:
    python trainer.py data.csv --format json
  
  Validate your data format:
    python trainer.py data.csv --validate
  
  Analyze last N days only:
    python trainer.py data.csv --days 14

REQUIRED CSV FORMAT:
  Minimum: date,total_sets
  Recommended: date,total_sets,average_rpe,sleep_quality_1_10
  
  Example:
    date,user_name,total_sets,average_rpe,sleep_quality_1_10
    2024-01-01,Alex,12,7.5,8
    2024-01-02,Alex,14,8.0,7
        """
    )
    
    # Positional arguments
    parser.add_argument('data_file', nargs='?', 
                       help='Training data CSV file path')
    
    # Action arguments
    parser.add_argument('--demo', action='store_true',
                       help='Run demonstration with synthetic data')
    parser.add_argument('--generate-data', type=int, metavar='DAYS',
                       help='Generate synthetic training data for N days')
    parser.add_argument('--validate', action='store_true',
                       help='Validate data format without making predictions')
    
    # Analysis options
    parser.add_argument('--user', type=str,
                       help='Filter analysis to specific user')
    parser.add_argument('--days', type=int, default=14,
                       help='Number of recent days to analyze (default: 14)')
    parser.add_argument('--format', choices=['table', 'json'], default='table',
                       help='Output format (default: table)')
    
    # Output options
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path (optional)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Reduce output verbosity')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    return parser

def run_demo():
    """Run demonstration with synthetic training data."""
    print("🎮 DEMO MODE: Generating synthetic training data...")
    
    try:
        # Generate sample data
        from ml.data.data_generator import AdvancedDataGenerator
        print("   → Creating user profiles...")
        generator = AdvancedDataGenerator()
        
        print("   → Simulating 30 days of training...")
        data = generator.generate_all_users_data(30)
        
        print(f"   → Generated {len(data)} training records")
        print(f"   → Users: {', '.join(data['user_name'].unique())}")
        
        # Demonstrate predictions for each user
        users = data['user_name'].unique()
        
        for user in users:
            print(f"\n{'='*60}")
            print(f"📊 ANALYZING: {user.upper()}")
            print('='*60)
            
            user_data = data[data['user_name'] == user].tail(14)
            make_and_display_prediction(user_data, user)
        
        print(f"\n{'='*60}")
        print("✅ DEMO COMPLETE!")
        print("💡 To analyze your own data: python trainer.py your_data.csv")
        print("💡 To generate training data: python trainer.py --generate-data 90")
        
    except ImportError as e:
        print(f"❌ Missing components for demo: {e}")
        print("💡 Make sure all ml.data components are available")
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def generate_data(days: int, output_file: Optional[str] = None):
    """Generate synthetic training data."""
    if days < 1 or days > 365:
        print("❌ Days must be between 1 and 365")
        return
    
    print(f"📊 Generating {days} days of synthetic training data...")
    
    try:
        from ml.data.data_generator import AdvancedDataGenerator
        
        print("   → Creating realistic user profiles...")
        generator = AdvancedDataGenerator()
        
        print("   → Simulating training sessions, sleep, recovery...")
        data = generator.generate_all_users_data(days)
        
        # Determine output filename
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            output_file = f"training_data_{days}days_{timestamp}.csv"
        
        # Save data
        data.to_csv(output_file, index=False)
        
        # Display summary
        print(f"\n✅ Data generated successfully!")
        print(f"   📁 File: {output_file}")
        print(f"   📊 Records: {len(data):,}")
        print(f"   👥 Users: {data['user_name'].nunique()}")
        print(f"   📈 Features: {len(data.columns)}")
        print(f"   📅 Date range: {data['date'].min()} to {data['date'].max()}")
        
        # Show muscle gain summary
        gain_stats = data.groupby('user_name')['muscle_gain_kg_per_week'].mean()
        print(f"\n💪 Average weekly muscle gains:")
        for user, gain in gain_stats.items():
            annual_gain = gain * 52
            print(f"   {user}: {gain:.3f} kg/week ({annual_gain:.1f} kg/year)")
        
        print(f"\n💡 To analyze this data: python trainer.py {output_file}")
        
    except ImportError as e:
        print(f"❌ Missing data generation components: {e}")
    except Exception as e:
        print(f"❌ Data generation failed: {e}")

def validate_data_file(data_file: str):
    """Validate training data file format."""
    if not os.path.exists(data_file):
        print(f"❌ File not found: {data_file}")
        return False
    
    print(f"🔍 Validating data file: {data_file}")
    
    try:
        # Load data
        data = pd.read_csv(data_file)
        
        # Basic validation
        print(f"   📊 Records: {len(data):,}")
        print(f"   📋 Columns: {len(data.columns)} ({', '.join(data.columns[:5])}...)")
        
        # Check required columns
        required_cols = ['total_sets']
        missing_required = [col for col in required_cols if col not in data.columns]
        
        if missing_required:
            print(f"❌ Missing required columns: {missing_required}")
            print(f"   Available columns: {list(data.columns)}")
            return False
        
        # Check optional columns
        recommended_cols = ['average_rpe', 'sleep_quality_1_10', 'date']
        missing_recommended = [col for col in recommended_cols if col not in data.columns]
        
        if missing_recommended:
            print(f"⚠️  Missing recommended columns: {missing_recommended}")
            print("   Predictions will use default values for missing features")
        
        # Data quality checks
        warnings = []
        
        # Check for sufficient data
        if len(data) < 7:
            warnings.append(f"Only {len(data)} records - need 7+ for reliable predictions")
        
        # Check for null values in required columns
        null_counts = data[required_cols].isnull().sum()
        for col, nulls in null_counts.items():
            if nulls > 0:
                warnings.append(f"Column '{col}' has {nulls} null values")
        
        # Check data ranges
        if 'total_sets' in data.columns:
            sets_range = (data['total_sets'].min(), data['total_sets'].max())
            if sets_range[0] < 0:
                warnings.append("Negative total_sets values found")
            if sets_range[1] > 50:
                warnings.append("Very high total_sets values (>50) found")
            print(f"   💪 Training volume range: {sets_range[0]}-{sets_range[1]} sets")
        
        if 'average_rpe' in data.columns:
            rpe_range = (data['average_rpe'].min(), data['average_rpe'].max())
            if rpe_range[0] < 1 or rpe_range[1] > 10:
                warnings.append("RPE values outside 1-10 range")
            print(f"   🔥 RPE range: {rpe_range[0]:.1f}-{rpe_range[1]:.1f}")
        
        # Display warnings
        if warnings:
            print("⚠️  Data quality warnings:")
            for warning in warnings:
                print(f"   - {warning}")
        else:
            print("✅ Data quality looks good!")
        
        # Check for users
        if 'user_name' in data.columns:
            users = data['user_name'].unique()
            print(f"   👥 Users found: {len(users)} ({', '.join(users[:3])}{'...' if len(users) > 3 else ''})")
        
        print("✅ Validation complete!")
        return True
        
    except pd.errors.EmptyDataError:
        print("❌ File is empty")
        return False
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

def analyze_data(data_file: str, user_filter: Optional[str], 
                output_format: str, days: int, output_file: Optional[str]):
    """Analyze training data and make predictions."""
    
    # Validate file exists
    if not os.path.exists(data_file):
        print(f"❌ File not found: {data_file}")
        print(f"💡 Make sure the file path is correct: {os.path.abspath(data_file)}")
        return
    
    print(f"📈 Loading training data: {data_file}")
    
    try:
        # Load data
        data = pd.read_csv(data_file)
        print(f"   ✅ Loaded {len(data):,} records with {len(data.columns)} features")
        
        # Parse dates if available
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            date_range = f"{data['date'].min().date()} to {data['date'].max().date()}"
            print(f"   📅 Date range: {date_range}")
        
        # Filter by user if specified
        if user_filter:
            if 'user_name' in data.columns:
                original_len = len(data)
                data = data[data['user_name'].str.contains(user_filter, case=False, na=False)]
                
                if len(data) == 0:
                    available_users = data['user_name'].unique() if 'user_name' in data.columns else []
                    print(f"❌ No data found for user matching '{user_filter}'")
                    if available_users:
                        print(f"   Available users: {', '.join(available_users)}")
                    return
                
                print(f"   🎯 Filtered to user(s) matching '{user_filter}': {len(data)} records")
            else:
                print("⚠️  No 'user_name' column found, ignoring user filter")
        
        # Use most recent N days
        if days > 0:
            data = data.tail(days)
            print(f"   📊 Using last {min(days, len(data))} days of data")
        
        # Make prediction
        result = make_and_display_prediction(data, user_filter or "Training Analysis")
        
        # Save output if requested
        if output_file:
            save_prediction_output(result, output_file, output_format)
        
    except pd.errors.EmptyDataError:
        print("❌ The CSV file is empty")
    except pd.errors.ParserError as e:
        print(f"❌ Error parsing CSV file: {e}")
        print("💡 Make sure the file is a valid CSV with proper headers")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("💡 Run with --validate to check your data format")

def make_and_display_prediction(data: pd.DataFrame, context_name: str):
    """Make prediction and display results."""
    
    try:
        # Import and use enhanced predictor
        from ml.models.enhanced_predictor import EnhancedLiftingPredictor
        
        print(f"   🧠 Making prediction using {len(data)} days of data...")
        predictor = EnhancedLiftingPredictor()
        result = predictor.predict(data)
        
        if result.success:
            display_prediction_results(result, context_name)
            return result
        else:
            print(f"❌ Prediction failed: {result.error}")
            print("💡 Try --validate to check your data format")
            return None
            
    except ImportError as e:
        print(f"❌ Missing prediction components: {e}")
        print("💡 Make sure all ml.models components are available")
        return None
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

def display_prediction_results(result, context_name: str):
    """Display prediction results in formatted table."""
    
    # Determine strength prediction emoji and message
    strength_icons = {
        "likely": "💪",
        "possible": "🤔", 
        "unlikely": "😐"
    }
    
    strength_messages = {
        "likely": "STRENGTH GAINS LIKELY this week!",
        "possible": "STRENGTH GAINS POSSIBLE this week",
        "unlikely": "STRENGTH GAINS UNLIKELY this week"
    }
    
    icon = strength_icons.get(result.prediction, "❓")
    message = strength_messages.get(result.prediction, f"Prediction: {result.prediction}")
    
    # Main results display
    print(f"\n🎯 PREDICTION RESULTS: {context_name.upper()}")
    print("=" * 70)
    print(f"{icon} {message}")
    print(f"   Confidence: {result.confidence:.0f}%")
    print("")
    
    # Recommendations section
    print("📋 TRAINING RECOMMENDATIONS:")
    print(f"   💪 RIR (Reps in Reserve): {result.rir_recommendation}")
    print(f"   ⏱️  Rest between sets: {result.rest_seconds} seconds")
    print("")
    
    # Reasoning section
    print(f"🧠 Analysis: {result.reason}")
    
    # Warnings section
    if result.warning:
        print(f"⚠️  Important notes: {result.warning}")
    
    print("=" * 70)
    
    # Training tips based on prediction
    tips = get_training_tips(result)
    if tips:
        print("💡 TRAINING TIPS:")
        for tip in tips:
            print(f"   • {tip}")
        print("")

def get_training_tips(result) -> list:
    """Get contextual training tips based on prediction."""
    tips = []
    
    if result.prediction == "likely":
        tips.append("Great conditions for strength gains - consider attempting PRs")
        tips.append("Focus on progressive overload this week")
        
    elif result.prediction == "possible": 
        tips.append("Decent training week ahead - maintain consistent effort")
        tips.append("Monitor recovery closely for next week's planning")
        
    elif result.prediction == "unlikely":
        tips.append("Consider backing off intensity this week")
        tips.append("Focus on recovery and technique refinement")
    
    # RIR-specific tips
    if result.rir_recommendation == "0-1 RIR":
        tips.append("You can push close to failure - good recovery detected")
    elif result.rir_recommendation == "3-4 RIR":
        tips.append("Keep more reps in reserve - prioritize recovery")
    
    # Rest-specific tips
    if result.rest_seconds >= 240:
        tips.append("Take longer rests to ensure full recovery between sets")
    elif result.rest_seconds <= 150:
        tips.append("Standard rest periods should be sufficient")
    
    return tips

def save_prediction_output(result, output_file: str, format_type: str):
    """Save prediction results to file."""
    try:
        if format_type == 'json':
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'prediction': result.prediction,
                'confidence': result.confidence,
                'rir_recommendation': result.rir_recommendation, 
                'rest_seconds': result.rest_seconds,
                'reason': result.reason,
                'warning': result.warning,
                'success': result.success
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
                
        else:  # text format
            with open(output_file, 'w') as f:
                f.write(f"Scientific Hypertrophy Trainer Prediction Report\n")
                f.write(f"Generated: {datetime.now()}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Prediction: {result.prediction}\n")
                f.write(f"Confidence: {result.confidence:.0f}%\n")
                f.write(f"RIR Recommendation: {result.rir_recommendation}\n")
                f.write(f"Rest Time: {result.rest_seconds} seconds\n")
                f.write(f"Reasoning: {result.reason}\n")
                if result.warning:
                    f.write(f"Warnings: {result.warning}\n")
        
        print(f"💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"⚠️  Failed to save output: {e}")

if __name__ == '__main__':
    main()
