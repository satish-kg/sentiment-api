import requests
import json

api_url = "http://127.0.0.1:8000/predict_sentiment"

# The text you want to classify
payload = {
    "text": "these is anti-incumbency sentiments against the current government!"
}

# Send the POST request
response = requests.post(api_url, data=json.dumps(payload), headers={"Content-Type": "application/json"})

if response.status_code == 200:
    print("API Response:", response.json())
else:
    print(f"Error: {response.status_code}, {response.text}")