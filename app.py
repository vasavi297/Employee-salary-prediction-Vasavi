from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model (should be a pipeline that includes preprocessing)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        company = request.form['company']
        job = request.form['job']
        degree = request.form['degree']
        experience = float(request.form['experience'])

        # Create DataFrame for prediction
        input_data = pd.DataFrame({
            'company': [company],
            'job': [job],
            'degree': [degree],
            'experience': [experience]
        })

        try:
            result = model.predict(input_data)[0]
            prediction = round(result, 2)
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
