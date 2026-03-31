<h1 align="center">Clinical Diabetes Risk Prediction</h1>

<p align="center">
  <strong>A Machine Learning Capstone Project for Early Risk Assessment</strong>
</p>

<hr>

<h2>Project Overview</h2>
<p>
  This project addresses the critical need for early diabetes detection by analyzing clinical health metrics and regional data. 
  By utilizing <b>Logistic Regression</b>, the system evaluates physiological factors—such as Glucose levels, BMI, and Age—to 
  provide a risk assessment. This helps healthcare providers in regions like <i>Sagar, Bhopal, and Indore</i> prioritize 
  high-risk patients for early intervention.
</p>

<h2>Dataset Description</h2>
<p>The project utilizes a multi-stage data approach across three primary files:</p>
<ul>
  <li><b>diabetes.csv:</b> The baseline clinical features (Pregnancies, Glucose, Blood Pressure, BMI, etc.).</li>
  <li><b>diabetes1.csv:</b> Used for initial exploratory data analysis (EDA) and testing missing value handling.</li>
  <li><b>diabetes2.csv:</b> An enhanced dataset including <b>Location</b> data to observe regional health trends.</li>
</ul>

<h2>Technical Implementation</h2>
<p>The solution is broken down into two main execution phases:</p>

<h3>1. Data Exploration (Mainpgm.py)</h3>
<p>
  Performs statistical analysis, correlation mapping, and initial preprocessing (handling null values and data sampling) 
  to understand the relationship between vitals and outcomes.
</p>

<h3>2. Clinical Modeling (LinearRegression.py)</h3>
<p>
  Despite the filename, this script implements a <b>Logistic Regression</b> classifier. Key features include:
</p>
<ul>
  <li><b>Data Validation:</b> Ensures required clinical columns are present before processing.</li>
  <li><b>Outlier Removal:</b> Filters out physiologically impossible "zero" values for BMI and Glucose.</li>
  <li><b>Model Training:</b> Uses a 70/30 stratified split to maintain class balance.</li>
  <li><b>Evaluation:</b> Outputs ROC-AUC scores and detailed Classification Reports.</li>
</ul>

<h2>Installation & Usage</h2>
<pre>
# Install dependencies
pip install pandas scikit-learn joblib

# Run data analysis
python Mainpgm.py

# Train and evaluate the model
python LinearRegression.py
</pre>

<h2>Project Results</h2>
<table border="1">
  <tr>
    <th>Metric</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><b>Model Type</b></td>
    <td>Logistic Regression (Balanced)</td>
  </tr>
  <tr>
    <td><b>Target Variable</b></td>
    <td>Outcome (0:
