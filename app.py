from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

@app.route('/')
def home():
    # This looks for index.html in the /templates folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker_input = request.json.get('ticker')
    if not ticker_input:
        return jsonify({"error": "No ticker provided"}), 400
        
    ticker = ticker_input.upper().strip()
    
    try:
        # 1. Download data (last 2 years)
        data = yf.download(ticker, period="2y", interval="1d")
        
        if data.empty:
            return jsonify({"error": "Stock symbol not found or no data available"}), 404

        # 2. AI Logic: Create Target (1 if price goes UP tomorrow, 0 if DOWN)
        # .shift(-1) looks at the next day's price
        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
        
        # 3. Features: Use current price and a 10-day Moving Average
        data["MA10"] = data["Close"].rolling(10).mean()
        data = data.dropna() # Remove empty rows
        
        X = data[["Close", "MA10"]]
        y = data["Target"]
        
        # 4. Train the Model (Random Forest)
        model = RandomForestClassifier(n_estimators=50, random_state=1)
        model.fit(X[:-1], y[:-1]) # Train on all days except the very last one
        
        # 5. Predict for "Tomorrow" using the most recent data row
        last_row = X.tail(1)
        prediction_array = model.predict(last_row)
        
        # --- FIX: Convert NumPy/Pandas types to standard Python types ---
        # prediction_array[0] is a numpy.int64, we convert it to a normal int
        prediction_val = int(prediction_array[0])
        
        # data['Close'].iloc[-1] is a pandas.Series/Float64, we convert to float
        current_price = float(data['Close'].iloc[-1])
        
        result_text = "HIGH 🚀" if prediction_val == 1 else "LESS 📉"
        
        return jsonify({
            "ticker": ticker,
            "prediction": result_text,
            "price": round(current_price, 2)
        })

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": "An error occurred while processing the data"}), 500

if __name__ == '__main__':
    # Setting debug=True helps you see errors in the terminal
    app.run(debug=True, port=5000)
