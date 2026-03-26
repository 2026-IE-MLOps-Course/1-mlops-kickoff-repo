import requests
import json

BASE_URL = "http://127.0.0.1:8000"

payload = {
    "customers": [
        {
            "customerID": "test-001",
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.0,
            "TotalCharges": "840"
        }
    ]
}

response = requests.post(f"{BASE_URL}/predict", json=payload)

print("Status code:", response.status_code)
print(json.dumps(response.json(), indent=2))
