 import pickle

# Assume you have a trained model called model
pickle.dump(model, open("model.pkl", "wb"))


from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the model
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return "Model is running successfully!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
