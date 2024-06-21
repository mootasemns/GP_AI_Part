from fastapi import FastAPI, Request
import logging
import uvicorn

# Initialize FastAPI app for alert listener
alert_app = FastAPI()

# Setup logging
logging.basicConfig(filename='logs/alerts.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s')

@alert_app.post("/alert")
async def receive_alert(request: Request):
    """Receive alert from POST request and log the message.

    Args:
        request (Request): The incoming request containing alert data.

    Returns:
        dict: Response status indicating the alert was received.
    """
    # Parse the incoming JSON request data
    alert_data = await request.json()
    # Log the received alert message
    logging.info(f"Received alert: {alert_data['message']}")
    # Return a response indicating the alert was received
    return {"status": "alert received"}

if __name__ == "__main__":
    # Run the FastAPI app using uvicorn
    uvicorn.run(alert_app, host="localhost", port=8000)
