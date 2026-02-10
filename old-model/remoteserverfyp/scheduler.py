import requests
from datetime import datetime

# The URL of your running Flask server
FLASK_SERVER_URL = "http://127.0.0.1:5000"

def check_and_trigger_retraining():
    """
    Checks if it's the first day of a new quarter and triggers retraining if true.
    This version is non-interactive, designed for automated execution (e.g., via cron).
    """
    today = datetime.now()
    
    # Check if it's the first day of a quarter (January, April, July, October)
    is_first_day_of_quarter = (today.day == 1) and (today.month in [1, 4, 7, 10])

    # Log the current time and the check being performed
    print(f"[{today.strftime('%Y-%m-%d %H:%M:%S')}] Scheduler running check.")
    
    # The decision is now fully automated based on the date
    if is_first_day_of_quarter:
        print("Condition met: It is the first day of a new quarter. Triggering retraining...")
        try:
            # Send a POST request to the /retrain endpoint
            response = requests.post(f"{FLASK_SERVER_URL}/retrain")
            response.raise_for_status()  # Raises an exception for bad status codes
            
            print("Retraining request sent successfully.")
            print("Server response:", response.json())
            
        except requests.exceptions.RequestException as e:
            print(f"Error: Could not connect to the server at {FLASK_SERVER_URL}.")
            print(f"Please ensure the Flask app ('app.py') is running.")
            print(f"Details: {e}")
    else:
        print("Condition for retraining not met. No action taken.")

if __name__ == "__main__":
    check_and_trigger_retraining()

