import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify

# Initialize Flask App
app = Flask(__name__)

# Load Dataset
df = pd.read_csv('diabetes.csv')

# Prepare Data for Model Training
X = df[['Glucose', 'BloodPressure', 'BMI', 'Age']]
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Function to generate glucose advice
def get_glucose_advice(glucose):
    if glucose < 100:
        return "Your glucose levels are normal. Keep up your healthy habits!"
    elif 100 <= glucose <= 140:
        return "Your glucose levels are slightly high. Consider a balanced diet and regular exercise."
    else:
        return "Your glucose levels are very high. Please reconsider your diet and exercise routine."

# API Route for Predictions
@app.route('/predict', methods=['POST'])
def predict_diabetes():
    try:
        # Get JSON Data from Request
        data = request.get_json()
        glucose = float(data['Glucose'])
        blood_pressure = float(data['BloodPressure'])
        bmi = float(data['BMI'])
        age = int(data['Age'])

        # Prepare Input Data
        input_data = [[glucose, blood_pressure, bmi, age]]
        input_df = pd.DataFrame(input_data, columns=['Glucose', 'BloodPressure', 'BMI', 'Age'])

        # Make Prediction
        prediction = model.predict(input_df)[0]
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"

        # Get Advice
        advice = get_glucose_advice(glucose)

        return jsonify({'prediction': result, 'advice': advice})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
