from flask import Flask, request, render_template, redirect
from onnxruntime import InferenceSession
from onnxruntime.datasets import get_example
import onnx
import yfinance as yfin
from pandas_datareader import data as pdr
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib.figure import Figure
import base64
from io import BytesIO

dates = {}
app = Flask(__name__)

@app.route('/form')
def form():
    #stock = ["IBM", "AAPL", "MSFT"]
    past_duration = ["6mo", "1y"]
    #future_duration = ["30","40"]
    return render_template('form.html', past_duration=past_duration)

@app.route('/data', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form
        dates = request.form
        app.logger.info(request.form)
        df = yfin.download(tickers='IBM', period=form_data['past_duration'])
        dataset = df['Close'].fillna(method='ffill')
        dataset = dataset.values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(dataset)
        dataset = scaler.transform(dataset)
        # generate the input and output sequences
        n_lookback = 60  # length of input sequences (lookback period)
        #n_forecast = int(form_data['future_duration'])  # length of output sequences (forecast period)
        n_forecast = 30
        X = []
        Y = []
        for i in range(n_lookback, len(dataset) - n_forecast + 1):
            X.append(dataset[i - n_lookback: i])
            Y.append(dataset[i: i + n_forecast])
        X = np.array(X)
        Y = np.array(Y)
        # generate the forecasts
        X_ = dataset[- n_lookback:]  # last available input sequence
        X_ = X_.reshape(1, n_lookback, 1)
        X_ = np.array(X_, dtype=np.float32)
        filename = "future_trend.onnx"
        sess = InferenceSession(filename)
        print("----")
        print(sess.get_inputs())
        print(sess.get_outputs())
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        result = sess.run([output_name], {input_name: X_})
        print("==")
        Y_ = np.array(result).reshape(-1, 1)
        Y_ = scaler.inverse_transform(Y_)
        predicted_stock_price = Y_
        # organize the results in a data frame
        df_past = df[['Close']].reset_index()
        df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
        df_past['Date'] = pd.to_datetime(df_past['Date'])
        df_past['Forecast'] = np.nan
        df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]
        df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
        df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
        df_future['Forecast'] = Y_.flatten()
        df_future['Actual'] = np.nan
        results = pd.concat([df_past, df_future])
        results = results.set_index('Date')
        # Generate the figure **without using pyplot**.
        fig = Figure()
        ax = fig.subplots()
        fig.suptitle("IBM")
        ax.plot(results)
        # Save it to a temporary buffer.
        buf = BytesIO()
        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f"<img src='data:image/png;base64,{data}'/>"
        #return render_template('data.html',form_data = form_data)

@app.route('/health', methods=['GET'])
def health():
    """Return service health"""
    return 'ok'

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port='7000')