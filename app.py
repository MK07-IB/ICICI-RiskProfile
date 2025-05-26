from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and encoders
model = joblib.load("best.model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

@app.route('/')
def home():
    return "ICICI Risk Profiling Flask App is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    df = pd.read_csv(file)

    for col in label_encoders:
        if col in df.columns:
            df[col] = label_encoders[col].transform(df[col].astype(str))

    preds = model.predict(df)
    df['Predicted_Risk_Profile'] = target_encoder.inverse_transform(preds)
    result = df.to_dict(orient='records')

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
