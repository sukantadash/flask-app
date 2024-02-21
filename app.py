from flask import Flask, request, render_template, redirect
from onnxruntime import InferenceSession
from onnxruntime.datasets import get_example
import onnx
import yfinance as yfin
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from matplotlib.figure import Figure
import base64
from io import BytesIO

dates = {}
app = Flask(__name__)

@app.route('/form')
def form():
    return render_template('form.html')

def print_data():
    for k,v in dates:
        print(f"{k} {v}")

@app.route('/data', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form
        dates = request.form
        app.logger.info(request.form)
        yfin.pdr_override()
        dataset_testing = pdr.get_data_yahoo('IBM', start=form_data['StartDate'], end=form_data['EndDate'])
        app.logger.info(dataset_testing)
        dataset_testing.reset_index()
        actual_stock_price = dataset_testing[['Open']].values
        inputs = actual_stock_price
        inputs = inputs.reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        inputs = scaler.fit_transform(inputs)
        app.logger.info(inputs)
        x_test = []
        for i in range(60, 81):
            x_test.append(inputs[i-60:i, 0])
        x_test = np.array(x_test, dtype=np.float32)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        #with open('predict_trend.onnx', 'rb') as f:
        #    model = onnx.load(f)
            #app.logger.info(model.summary())
        #onnx.checker.check_model(model)
        #predictions = model.predict(x_test)
        #sess = rt.InferenceSession(model)
        #input_name = sess.get_inputs()[0].name
        #label_name = sess.get_outputs()[0].name
        #predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
        #example1 = get_example("sigmoid.onnx")
        #sess = rt.InferenceSession(model, providers=rt.get_available_providers())
        #input_name = sess.get_inputs()[0].name
        #print("input name", input_name)
        #input_shape = sess.get_inputs()[0].shape
        #print("input shape", input_shape)
        #input_type = sess.get_inputs()[0].type
        #print("input type", input_type)
        filename = "predict_trend.onnx"
        sess = InferenceSession(filename)
        #x_test, y_test = json_to_ndarray()
        print("----")
        print(sess.get_inputs())
        print(sess.get_outputs())
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        result = sess.run([output_name], {input_name: x_test})
        print("==")
        predicted_stock_price = np.array(result)
        predicted_stock_price = predicted_stock_price.reshape(-1, predicted_stock_price.shape[-1])
        print(predicted_stock_price)
        predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
        print(predicted_stock_price)
        print("==actual")
        print(actual_stock_price)
        # Generate the figure **without using pyplot**.
        fig = Figure()
        ax = fig.subplots()
        #ax.plot([1, 2])
        ax.plot(actual_stock_price)
        ax.plot(predicted_stock_price)
        # Save it to a temporary buffer.
        buf = BytesIO()
        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f"<img src='data:image/png;base64,{data}'/>"
        #return render_template('data.html',form_data = form_data)

@app.route('/calculate')
def calculate():
    print(dates)
    form_data = dates
    return render_template('data.html',form_data = form_data)



    
#app.run(host='localhost', port=5000)



"""# Load model
with open('model.onnx', 'rb') as f:
    model = onnx.load(f)
onnx.checker.check_model(model)

model_name = "Time to predict the Stock Trend"
model_file = 'model.onnx'
version = "v1.0.0"
"""

@app.route('/user/<name>')
def user(name):
    return render_template('user.html', name=name)

@app.route('/info', methods=['GET'])
def info():
    """Return model information, version how to call"""
    result = {}

    result["name"] = model_name
    result["version"] = version

    return result


@app.route('/health', methods=['GET'])
def health():
    """REturn service health"""
    return 'ok'


@app.route('/predict', methods=['POST'])
def predict():
    
    feature_dict = request.get_json()
    if not feature_dict:
        return {
            'error': 'Body is empty.'
        }, 500

    try:
        return {
            'status': 200,
            'prediction': "hello"
            }
    except ValueError as e:
        return {'error': str(e).split('\n')[-1].strip()}, 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    print_data()