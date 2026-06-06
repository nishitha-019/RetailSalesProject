from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("sales_data.csv")

# Prepare input and output data
X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Home page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction page
@app.route("/predict", methods=["POST"])
def predict():

    tv = float(request.form["tv"])
    radio = float(request.form["radio"])
    newspaper = float(request.form["newspaper"])

    result = model.predict([[tv, radio, newspaper]])

    return render_template(
        "result.html",
        tv=tv,
        radio=radio,
        newspaper=newspaper,
        prediction=round(result[0], 2)
    )

# Run the application
if __name__ == "__main__":
    app.run(debug=True)