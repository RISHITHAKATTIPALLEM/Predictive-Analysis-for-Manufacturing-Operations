from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

app = FastAPI()

# In-memory storage for the uploaded dataset and model
data_storage = {"dataset": None, "model": None}

@app.post("/upload")
def upload_data(file: UploadFile = File(...)):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file.file)
        data_storage["dataset"] = df
        return {"message": "Dataset uploaded successfully", "columns": df.columns.tolist()}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/train")
def train_model():
    if data_storage["dataset"] is None:
        return JSONResponse(status_code=400, content={"error": "No dataset uploaded"})

    try:
        # Prepare the dataset
        df = data_storage["dataset"]
        if "Downtime_Flag" not in df.columns:
            return JSONResponse(status_code=400, content={"error": "Dataset must contain 'Downtime_Flag' column"})

        X = df.drop(columns=["Downtime_Flag"])
        y = df["Downtime_Flag"]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')

        # Save the model
        joblib.dump(model, "model.pkl")
        data_storage["model"] = model

        return {"message": "Model trained successfully", "accuracy": accuracy, "f1_score": f1}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/predict")
def predict(input_data: dict):
    if data_storage["model"] is None:
        return JSONResponse(status_code=400, content={"error": "No model trained"})

    try:
        # Load the model
        model = joblib.load("model.pkl")

        # Create a DataFrame for prediction
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        confidence = np.max(model.predict_proba(input_df))

        return {"Downtime": "Yes" if prediction == 1 else "No", "Confidence": round(confidence, 2)}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")

