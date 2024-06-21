import os
import time
import requests
import torch
from torchvision import transforms
from PIL import ImageGrab  
from utils import CheatDetectionModel 
from utils import readConfig
import mlflow
from mlflow import log_artifact

# Load configuration files
model_config_file = readConfig("config/model_config.yaml")
deployment_config_file = readConfig("config/deployment_config.yaml")
deployment_config = deployment_config_file["DEPLOYMENT"]
target_website_url = deployment_config["target_website_url"]
sshot_delay_time = deployment_config["sshot_delay_time"]

model_config = model_config_file["MODEL"]
num_classes = model_config["num_classes"]
model_path = model_config["model_path"]

# Load model
model = CheatDetectionModel(num_classes=num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# Define image transformations
data_transforms = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image_class(image):
    """Predict the class of the given image using the loaded model.

    Args:
        image (PIL.Image): Image to be classified.

    Returns:
        int: Predicted class label.
    """
    image = data_transforms(image)
    image = torch.unsqueeze(image, 0)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

def send_alert():
    """Send an alert to the target website URL indicating cheating is detected."""
    url = target_website_url
    data = {'message': 'Cheating detected'}
    requests.post(url, json=data)

print("Starting the loop")

# Start MLflow run
mlflow.start_run()

while True:
    print("Taking Screenshot ...")
    # Take a screenshot
    screenshot = ImageGrab.grab()
    
    # Log screenshot as an artifact to MLflow
    screenshot_path = "screenshot.png"
    screenshot.save(screenshot_path)
    log_artifact(screenshot_path)
    
    # Predict using the model
    prediction = predict_image_class(screenshot)
    print("Prediction:", prediction)
    
    # Log prediction result to MLflow
    mlflow.log_metric("prediction", prediction)
    
    # If cheating is detected, send an alert
    if prediction == 1:
        send_alert()
    
    # Wait for a few seconds before taking the next screenshot
    time.sleep(sshot_delay_time)

# End MLflow run
mlflow.end_run()
