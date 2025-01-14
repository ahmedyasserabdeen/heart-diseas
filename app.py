from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.io as pio
import joblib
import numpy as np
app = Flask(__name__)

# Load your dataset (make sure the path is correct)
df = pd.read_csv('heart.csv')  # Update this path with your dataset

# Example route to display form and process input data
@app.route('/', methods=['GET', 'POST'])
def index():
    # Visualizations
    fig_age_dist = create_age_distribution(df)
    fig_chol_dist = create_chol_distribution(df)
    fig_sex_distribution = create_sex_distribution(df)
    box_plot = create_boxplot(df)

    total_records = len(df)
    heart_disease_yes = len(df[df['target'] == 1])
    heart_disease_no = len(df[df['target'] == 0])
    males = len(df[df['sex'] == 1])
    females = len(df[df['sex'] == 0])
    
    # Default value for prediction
    prediction = None
    
    if request.method == 'POST':
        # Retrieve user input from the form
        age = request.form['age']
        sex = request.form['sex']
        cp = request.form['cp']
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form['fbs']
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form['exang']
        oldpeak = request.form['oldpeak']
        slope = request.form['slope']
        ca = request.form['ca']
        thal = request.form['thal']
        
        # Prediction (replace with actual model prediction)
        prediction = predict_heart_attack(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
        print()

    return render_template('index.html', 
                           prediction=prediction, 
                           fig_age_dist=fig_age_dist, 
                           fig_chol_dist=fig_chol_dist,
                           fig_sex_distribution=fig_sex_distribution, 
                           box_plot=box_plot, 
                           total_records=total_records, 
                           heart_disease_yes=heart_disease_yes,
                           heart_disease_no=heart_disease_no, 
                           males=males, 
                           females=females)

    

def predict_heart_attack(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    # Load the pre-trained model and preprocessor
    model = joblib.load('voting.joblib')  # Save your trained model as rf_model.pkl
    preprocessor = joblib.load('scaler.joblib')

    features=[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    data_array = np.array(features).reshape(1, -1)
    preprocessed_data = preprocessor.transform(data_array)
    prediction = model.predict(preprocessed_data)
    if prediction == 0:
        return "Heart Attack Risk: Low"
    elif prediction == 1:
        return "Heart Attack Risk: High"
 

def create_age_distribution(df):
    fig = px.histogram(df, x="age")
    fig.update_traces( marker_line_width=1)

    return pio.to_html(fig, full_html=False)

def create_chol_distribution(df):
    fig = px.histogram(df, x="chol")
    fig.update_traces( marker_line_width=1)

    return pio.to_html(fig, full_html=False)

def create_sex_distribution(df):
    fig = px.histogram(df, x="sex", color="target", barmode="group")
    return pio.to_html(fig, full_html=False)


def create_boxplot(df):
    fig = px.box(df, y=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])

    return pio.to_html(fig, full_html=False)




if __name__ == "__main__":
    app.run(debug=True)
