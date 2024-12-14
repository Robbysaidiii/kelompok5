from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle

# Load the trained model
with open('naive_bayes_model_fix.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict_rain():
    try:
        # Parse JSON input
        data = request.get_json()
        tn = float(data['tn'])
        tx = float(data['tx'])
        tavg = float(data['tavg'])
        rh_avg = float(data['rh_avg'])
        ss = float(data['ss'])
        ff_avg = float(data['ff_avg'])

        # Create DataFrame for prediction
        input_data = pd.DataFrame({
            'Tn': [tn],
            'Tx': [tx],
            'Tavg': [tavg],
            'RH_avg': [rh_avg],
            'ss': [ss],
            'ff_avg': [ff_avg]
        })

        # Make prediction
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            result = "Ya"
            message = "Cuaca sedang hujan nih, jangan lupa sedia payung dan jas hujan ya. Selamat beraktivitas!"
        else:
            result = "Tidak"
            message = "Cuaca sedang tidak hujan, silahkan beraktivitas tanpa khawatir kehujanan ya. Have fun!"

        # Return the result as JSON
        return jsonify({
            'success': True,
            'prediction': result,
            'message': message
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Route to render the index.html page
@app.route('/')
def index():
    return render_template('base.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
