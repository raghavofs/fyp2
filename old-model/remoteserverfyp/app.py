import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime

# Import the retraining function
from model_logic import train_and_baseline

app = Flask(__name__)

DATA_FILE = 'waterdata.csv'
# Approximately 90 days of readings (assuming readings every 3 days in the original data)
# Adjust this number based on your actual data frequency to match 3 months.
ROWS_TO_REMOVE_PER_QUARTER = 30 

@app.route('/add_reading', methods=['POST'])
def add_reading():
    """
    API Endpoint to add a new daily reading.
    It adds the new row and removes the oldest row to maintain the dataset size.
    """
    new_data = request.json
    if not new_data:
        return jsonify({"status": "error", "message": "No data provided"}), 400

    try:
        df = pd.read_csv(DATA_FILE)

        # Remove the first (oldest) row of data
        df = df.iloc[1:]

        # Append the new data
        # We create a new DataFrame from the new data to ensure columns align
        new_row_df = pd.DataFrame([new_data])
        df = pd.concat([df, new_row_df], ignore_index=True)
        
        # Save the updated dataframe back to the CSV
        df.to_csv(DATA_FILE, index=False)
        
        return jsonify({"status": "success", "message": f"Data for {new_data.get('Date')} added."}), 201

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/retrain', methods=['POST'])
def trigger_retraining():
    """
    API Endpoint to manually trigger the model retraining process.
    """
    print("\n[API] Received request to retrain the model.")
    try:
        success = train_and_baseline()
        if success:
            return jsonify({"status": "success", "message": "Model retraining completed successfully."}), 200
        else:
            return jsonify({"status": "error", "message": "Model retraining failed. Check server logs."}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=True)
