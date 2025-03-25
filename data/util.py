import pandas as pd
import argparse
from datetime import datetime

def process_hourly_data(input_file, output_file, hour='12:00:00'):
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        
        # Convert time column to datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # Create date and time columns
        df['date'] = df['time'].dt.date
        df['time_of_day'] = df['time'].dt.time
        
        # Convert hour parameter to time
        target_time = datetime.strptime(hour, '%H:%M:%S').time()
        
        # Filter rows for specified hour
        daily_data = df[df['time_of_day'] == target_time]
        
        # Drop the extra columns we created
        daily_data = daily_data.drop(['date', 'time_of_day'], axis=1)
        
        # Save to CSV
        daily_data.to_csv(output_file, index=False)
        
        print(f"Processing completed successfully!")
        print(f"Input records: {len(df)}")
        print(f"Output records: {len(daily_data)}")
        print(f"\nSample of processed data:")
        print(daily_data.head())
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process hourly data to daily data at specific hour')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('output_file', help='Output CSV file path')
    parser.add_argument('--hour', default='12:00:00', help='Hour to select (format: HH:MM:SS)')
    
    args = parser.parse_args()
    
    process_hourly_data(args.input_file, args.output_file, args.hour)