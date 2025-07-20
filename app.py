from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Mapping dictionaries
company_mapping = {
    "Google": 0,
    "Microsoft": 1,
    "Facebook": 2,
    "Amazon": 3,
    "Netflix": 4,
    "Apple": 5
}

job_mapping = {
    "Data Scientist": 0,
    "Software Engineer": 1,
    "Web Developer": 2,
    "Machine Learning Engineer": 3,
    "Data Analyst": 4
}

degree_mapping = {
    "Bachelors": 0,
    "Masters": 1,
    "PhD": 2
}

# Dummy average salary values for the chart
average_salaries = {
    "Data Scientist": 1800000,
    "Software Engineer": 1500000,
    "Web Developer": 1200000,
    "Machine Learning Engineer": 2000000,
    "Data Analyst": 1300000
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        company = request.form["company"]
        job = request.form["job"]
        degree = request.form["degree"]
        experience = int(request.form["experience"])

        # Create DataFrame for prediction
        features = pd.DataFrame([[company, job, degree, experience]],
                                columns=["company", "job", "degree", "experience"])

        # Predict salary
        prediction = model.predict(features)[0]

        # Send data to result template
        return render_template("result.html",
                               salary=int(prediction),
                               company=company,
                               job=job,
                               degree=degree,
                               experience=experience,
                               jobs=list(average_salaries.keys()),
                               values=list(average_salaries.values()))

    return render_template("index.html")

@app.route("/result")
def result():
    return render_template("result.html",
                           salary=None,
                           jobs=list(average_salaries.keys()),
                           values=list(average_salaries.values()))
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


