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
        return "- Your glucose levels are normal. Keep up your healthy habits!"
    elif 100 <= glucose <= 140:
        return "- Your glucose levels are slightly high. Consider a balanced diet and regular exercise."
    else:
        return "- Your glucose levels are very high. Please reconsider your diet and exercise routine."

def get_bmi_advice(bmi_value):
    if bmi_value < 18.5:
        return "\n- You are underweight. Consider incorporating a balanced diet with nutrient-rich foods to maintain a healthy weight. Stay active and take care of your well-being!"
    elif 18.5 <= bmi_value < 24.9:
        return "\n- Great job! You have a healthy weight. Keep up a balanced diet and regular exercise to maintain your well-being!"
    else:
        return "\n- You are in the overweight range. Focusing on a balanced diet and regular physical activity can help you achieve a healthier weight. Small, consistent changes can make a big difference!"

def get_bp_advice(blood_pressure):
    if blood_pressure >= 80:
        return "\n- You have high blood pressure. Ways to combat high blood pressure is to eat a healthy diet low in sodium, maintaining a good BMI, getting regular physical activity, limiting alcohol intake, managing stress, and getting enough sleep"
    else:
        return "\n- You have normal blood pressure. You are doing great! Keep on going! Keep maintainng a healthy low sodium diet and getting a regular exercise"

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
        result = (
            "Your results indicate a high likelihood of diabetes. This is not a medical diagnosis, but it suggests further evaluation. Consider consulting a healthcare professional for confirmation."
            if prediction == 1
            else
            "Your results suggest a low likelihood of diabetes. This does not guarantee you are free from risk. Maintaining a balanced diet, regular exercise, and health checkups is still recommended."
        )

        # Get Advice
        advice = get_glucose_advice(glucose) + get_bmi_advice(bmi) + get_bp_advice(blood_pressure)

        return jsonify({'prediction': result, 'advice': advice})
    
    except Exception as e:
        return jsonify({'error': str(e)})

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
