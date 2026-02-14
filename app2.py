from flask import Flask, jsonify
import joblib
import numpy as np
from supabase import create_client
import threading
import time
from dotenv import load_dotenv
import os


# -------------------- CONFIG --------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

THRESHOLD = 0.7
POLLING_INTERVAL = 15  # seconds

# -------------------- INIT --------------------

app = Flask(__name__)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

model = joblib.load("emergency_model.pkl")
scaler = joblib.load("scaler.pkl")

latest_result = {
    "probability": 0,
    "status": "UNKNOWN"
}

# -------------------- BACKGROUND POLLING --------------------

def poll_supabase():

    global latest_result

    while True:
        try:
            response = supabase.table("vulnerable_kit") \
                .select("*") \
                .order("created_at", desc=True)\
                .limit(1) \
                .execute()

            if response.data:
                print("RAW RESPONSE:", response.data)
                row = response.data[0]
                print("ROW:", row)

                sample = [[
                    row["spo2"],
                    (row["bodytemp"]* 9/5) + 32,
                    row["heartrate"],
                    row["stepcount"],
                    row["env_pressure"]
                ]]

                scaled = scaler.transform(sample)
                prob = model.predict_proba(scaled)[0][1]

                status = "EMERGENCY" if prob > THRESHOLD else "NORMAL"

                latest_result = {
                    "probability": round(float(prob), 3),
                    "status": status
                }

                print("Updated:", latest_result)

        except Exception as e:
            print("Supabase Error:", e)

        time.sleep(POLLING_INTERVAL)

# Start polling thread
threading.Thread(target=poll_supabase, daemon=True).start()


# -------------------- ROUTES --------------------

@app.route("/")
def home():
    return "ML Server Running"


@app.route("/status")
def status():
    return jsonify(latest_result)


# -------------------- RUN --------------------

if __name__ == "__main__":
    app.run(debug=True)
