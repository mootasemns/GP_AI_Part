import torch
from torchvision import transforms
from PIL import ImageGrab  # For capturing screenshots
from utils import CheatDetectionModel 
import requests
from fastapi import FastAPI
from threading import Thread
import uvicorn
from utils import readConfig

# Initialize FastAPI app
app = FastAPI()

# Read configuration files
model_config_file = readConfig('config/model_config.yaml')
deployment_config_file = readConfig('config/deployment_config.yaml')

# Extract health check configuration
health_check_config = deployment_config_file["HEALTH_CHECK"]
# Extract deployment configuration
deployment_config = deployment_config_file["DEPLOYMENT"]
# Extract model configuration
model_config = model_config_file["MODEL"]

# Model path and number of classes
model_path = model_config["model_path"]
num_classes = model_config["num_classes"]

# Host and port for health check
host = health_check_config["host"]
port = health_check_config["port"]

# Target website URL for sending alerts
target_website_url = deployment_config["target_website_url"]

@app.get("/health")
async def health_check():
    """Health check endpoint

    Returns:
        dict: Status of the service
    """
    return {"status": "running"}

# Load the model and set it to evaluation mode
model = CheatDetectionModel(num_classes=num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# Define transforms for input data
data_transforms = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image_class(image):
    """Predict the class of the given image

    Args:
        image (PIL.Image): Input image

    Returns:
        int: Predicted class label
    """
    image = data_transforms(image)
    image = torch.unsqueeze(image, 0)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

def send_alert():
    """Send an alert to the target website"""
    url = target_website_url  # Replace with the actual URL
    data = {'message': 'Cheating detected'}
    requests.post(url, json=data)

def run_fastapi():
    """Run the FastAPI application"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    # Start the FastAPI app
    run_fastapi()