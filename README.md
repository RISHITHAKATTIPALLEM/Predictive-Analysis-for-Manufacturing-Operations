# Predictive API for Manufacturing Operations

This project implements a RESTful API for predicting machine downtime or production defects using a manufacturing dataset. It includes endpoints to upload data, train a machine learning model, and make predictions.

---

## Features

- **Upload Endpoint (`/upload`)**: Accepts a CSV file containing manufacturing data (e.g., `Machine_ID`, `Temperature`, `Run_Time`).
- **Train Endpoint (`/train`)**: Trains a Logistic Regression model on the uploaded dataset and provides performance metrics like accuracy and F1 score.
- **Predict Endpoint (`/predict`)**: Accepts JSON input (e.g., `{"Temperature": 80, "Run_Time": 120}`) and returns predictions (e.g., `{"Downtime": "Yes", "Confidence": 0.85}`).
- **Interactive API Documentation**: Accessible via Swagger UI at `/docs`.

---

## Requirements

- Python 3.8+
- Installed libraries:
  ```bash
  pip install fastapi uvicorn scikit-learn pandas numpy joblib python-multipart
  ```

---

## Installation and Setup

1. **Clone the Repository**:

   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Install Dependencies**:

   ```bash
   pip install fastapi uvicorn scikit-learn pandas numpy joblib python-multipart
   ```

3. **Run the Application**:

   ```bash
   uvicorn main:app --reload
   ```

4. **Access the Application**:

   - Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the interactive Swagger UI.
   - The root endpoint (`/`) redirects to `/docs`.

---

## Endpoints

### 1. **Upload Endpoint**

- **URL**: `/upload`
- **Method**: POST
- **Description**: Upload a CSV file containing manufacturing data.
- **Input**:
  - File upload with columns like `Machine_ID`, `Temperature`, `Run_Time`, `Downtime_Flag`.
- **Response**:
  ```json
  {
    "message": "Dataset uploaded successfully",
    "columns": ["Machine_ID", "Temperature", "Run_Time", "Downtime_Flag"]
  }
  ```

### 2. **Train Endpoint**

- **URL**: `/train`
- **Method**: POST
- **Description**: Train a Logistic Regression model on the uploaded dataset.
- **Response**:
  ```json
  {
    "message": "Model trained successfully",
    "accuracy": 0.92,
    "f1_score": 0.88
  }
  ```

### 3. **Predict Endpoint**

- **URL**: `/predict`
- **Method**: POST
- **Description**: Predict downtime based on input JSON data.
- **Input**(Anything based you need):
  ```json
  {
    "Temperature": 80,
    "Run_Time": 120
  }
  ```

---

## Example Workflow

1. Start the API server:
   ```bash
   uvicorn main:app --reload
   ```
2. Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
3. Use `/upload` to upload a dataset.
4. Use `/train` to train the model.
5. Use `/predict` to make predictions based on input values.

