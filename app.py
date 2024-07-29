from flask import Flask, request, render_template, send_from_directory, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os
import logging

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

static_dir = os.path.join(os.path.dirname(__file__), 'static')
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Load and prepare the data
try:
    data = pd.read_csv('Iris.csv')
    X = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
    y = data["Species"].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the logistic regression model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Create and save visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x="SepalLengthCm", y="PetalLengthCm", hue="Species")
    plt.title("Iris Species - Sepal Length vs Petal Length")
    plt.savefig(os.path.join(static_dir, 'scatter_plot.png'))
    plt.close()

except Exception as e:
    app.logger.error(f"Error during data loading or model training: {str(e)}")
    model = None

@app.route('/')
def home():
    return render_template('index.html', prediction='')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction="Model not available")
    
    try:
        features = [float(request.form[f]) for f in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        prediction = model.predict(np.array(features).reshape(1, -1))[0]
    except ValueError:
        prediction = "Invalid input"
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        prediction = "Error occurred during prediction"
    
    return render_template('index.html', prediction=prediction)

@app.route('/report')
def report():
    if model is None:
        return jsonify({"error": "Model not available"})
    
    try:
        y_pred = model.predict(X_test)
        classification_report = metrics.classification_report(y_test, y_pred, output_dict=True)
        confusion_matrix = metrics.confusion_matrix(y_test, y_pred).tolist()
        accuracy = metrics.accuracy_score(y_test, y_pred)
        
        return jsonify({
            'classification_report': classification_report,
            'confusion_matrix': confusion_matrix,
            'accuracy': accuracy
        })
    except Exception as e:
        app.logger.error(f"Error generating report: {str(e)}")
        return jsonify({"error": "Error generating report"})

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(static_dir, filename)

if __name__ == '__main__':
    app.run(debug=True)