import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# CoinMarketCap API configuration
API_KEY = "e3c86887-3bfb-4a33-97c1-c7e908b37ce5"
BASE_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
HEADERS = {
    "Accepts": "application/json",
    "X-CMC_PRO_API_KEY": API_KEY,
}

# Fetch the top 150 cryptocurrencies
def fetch_cryptos():
    params = {
        "start": "1",
        "limit": "150",
        "convert": "USD",
    }
    response = requests.get(BASE_URL, headers=HEADERS, params=params)
    data = response.json()
    return data['data']

# Moving Averages (Simple MA and Exponential EMA)
def calculate_moving_averages(prices, period):
    prices = np.array(prices)
    sma = np.convolve(prices, np.ones(period)/period, mode='valid')
    ema = []
    multiplier = 2 / (period + 1)
    for i in range(len(prices)):
        if i == 0:
            ema.append(prices[i])
        else:
            ema.append((prices[i] - ema[i-1]) * multiplier + ema[i-1])
    return sma, ema

# Predict next week's ROI using Fibonacci retracement levels
def calculate_fibonacci_levels(prices):
    high = max(prices)
    low = min(prices)
    diff = high - low
    levels = {
        "0%": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "100%": low,
    }
    return levels

# Generate forecasts and insights
def analyze_crypto_data(crypto):
    historical_prices = crypto['quote']['USD']['price'] * (1 + np.random.normal(0, 0.05, 30))  # Simulated historical prices
    sma, ema = calculate_moving_averages(historical_prices, period=7)
    fib_levels = calculate_fibonacci_levels(historical_prices)
    
    roi_forecast = (ema[-1] - historical_prices[-1]) / historical_prices[-1] * 100
    return {
        "name": crypto['name'],
        "symbol": crypto['symbol'],
        "price": crypto['quote']['USD']['price'],
        "sma": sma,
        "ema": ema,
        "fibonacci_levels": fib_levels,
        "roi_forecast": roi_forecast,
    }

# Main script
cryptos = fetch_cryptos()
crypto_analysis = [analyze_crypto_data(crypto) for crypto in cryptos]

# Include all cryptocurrencies with ROI forecast
detailed_cryptos = [
    {
        "name": data["name"],
        "price": data["price"],
        "roi_forecast": data["roi_forecast"],
        "days": 7,
        "amount_to_buy": 100 / data["price"],
    }
    for data in crypto_analysis
]

# Save to CSV
df = pd.DataFrame(detailed_cryptos)
csv_filename = "crypto_recommendations.csv"
df.to_csv(csv_filename, index=False)

# Plot graphs
for crypto in detailed_cryptos[:5]:
    name = crypto['name']
    plt.figure()
    plt.title(f"{name} - Moving Averages and Fibonacci Levels")
    plt.plot(crypto_analysis[0]['sma'], label='SMA')
    plt.plot(crypto_analysis[0]['ema'], label='EMA')
    plt.axhline(crypto_analysis[0]['fibonacci_levels']['50%'], color='red', linestyle='--', label='Fibonacci 50%')
    plt.legend()
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.show()

print(f"Analysis completed. Results saved in {csv_filename}.")