import requests
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
import plotly.graph_objs as go
import plotly.express as px
import plotly
import json


app = Flask(__name__)

# Step 1: Fetch Data
def fetch_top_cryptos():
    endpoint = "https://api.binance.com/api/v3/ticker/24hr"
    response = requests.get(endpoint)
    data = response.json()
    
    # Sort by quoteVolume (24h trading volume) and get top 50
    sorted_data = sorted(data, key=lambda x: float(x['quoteVolume']), reverse=True)[:50]
    return sorted_data

# Step 2: Fetch Historical Data
def fetch_historical_data(symbol):
    endpoint = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=24"
    response = requests.get(endpoint)
    data = response.json()
    return data

def fetch_logo_url(crypto_name):
    url = f"https://www.google.com/search?q={crypto_name}+logo"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract the logo URL from the Google search results
        logo_url = soup.find('img')['src']
        return logo_url
    else:
        return None

# Step 3: Process Data
def process_data():
    top_cryptos = fetch_top_cryptos()
    processed_data = []
    for item in top_cryptos:
        symbol = item["symbol"]
        price = float(item["lastPrice"])
        price_change_24h = float(item["priceChangePercent"])
        volume_24h = float(item["volume"])
        
        # Fetch historical data for the graph
        historical_data = fetch_historical_data(symbol)
        timestamps = [x[0] for x in historical_data]
        prices = [float(x[4]) for x in historical_data]  # closing prices
        
        processed_data.append({
            "symbol": symbol,
            "price": price,
            "price_change_24h": price_change_24h,
            "volume_24h": volume_24h,
            "timestamps": timestamps,
            "prices": prices
        })
        
    return processed_data

# Step 4: Display Data
@app.route('/')
def index():
    data = process_data()
    tables = []
    graphs = []
    
    for crypto in data:
        tables.append({
            "symbol": crypto["symbol"],
            "price": crypto["price"],
            "price_change_24h": crypto["price_change_24h"],
            "volume_24h": crypto["volume_24h"]
        })
        
        color = 'green' if crypto["price_change_24h"] > 0 else 'red'
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=crypto["timestamps"],
            y=crypto["prices"],
            mode='lines',
            line=dict(color=color)
        ))
        fig.update_layout(
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(showgrid=False, visible=False),
            margin=dict(l=0, r=0, t=0, b=0),
            height=70,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        graphs.append({"symbol": crypto["symbol"], "graph": graph_json})
        
    return render_template('index.html', tables=tables, graphs=graphs)

@app.route('/search', methods=['POST'])
def search():
    symbol = request.form['symbol'].upper()
    return redirect(f"https://www.tradingview.com/symbols/{symbol}/")

if __name__ == "__main__":
    app.run(debug=True)
