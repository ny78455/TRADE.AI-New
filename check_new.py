import pandas as pd
import time
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
import os

def main():
    # Initialize Alpaca API
    load_dotenv()
    API_KEY = os.getenv('APCA_API_KEY_ID')
    API_SECRET = os.getenv('APCA_API_SECRET_KEY')
    BASE_URL = 'https://paper-api.alpaca.markets'

    # Initialize Alpaca API
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')
    # Path to your CSV file
    csv_path = 'artifacts/predictions_new.csv'
    
    while True:
        try:
            # Read CSV file
            df = pd.read_csv(csv_path, index_col='Datetime', parse_dates=True)
            
            # Get the latest row
            latest_row = df.iloc[-1]
            
            # Get the prediction signal
            signal = latest_row['Prediction']
            
            # Implement trading logic based on signal
            if signal == 1:
                # Execute sell order
                api.submit_order(
                    symbol='AVAX/USD',
                    qty=1,  # Adjust quantity as per your strategy
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                print("Executed sell order.")
            elif signal in [0,2, 3, 4, 5]:
                # Implement logic to sell after signal minutes
                minutes_to_wait = signal+1
                print(f"Uptrend signal detected. Will sell after {minutes_to_wait} minutes.")
                time.sleep(minutes_to_wait * 60)  # Wait for the specified minutes
                # Execute sell order after waiting
                api.submit_order(
                    symbol='AVAX/USD',
                    qty=1,  # Adjust quantity as per your strategy
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                print(f"Executed sell order after {minutes_to_wait} minutes.")
            
            # Wait for 8 seconds before checking again
            time.sleep(8)
        
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            time.sleep(8)  # Wait and retry

if __name__ == "__main__":
    main()
