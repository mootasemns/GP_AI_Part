import pyautogui
import time
import os
import webbrowser
from urllib.parse import quote_plus
from utils import readConfig

# Load data collection config
data_collection_config = readConfig('config/data_collection_config.yaml')

def take_screenshots(url, site_name, num_images):
    """Take screenshots of a given URL and save them to a specified folder.

    Args:
        url (str): URL of the website to take screenshots of.
        site_name (str): Name of the site, used for folder and file naming.
        num_images (int): Number of screenshots to take.
    """
    folder_path = f"data/{site_name}"
    os.makedirs(folder_path, exist_ok=True)
    
    # Specify browser and open the URL
    webbrowser.open(url)

    time.sleep(5)  # Adjust this wait time as needed for page load
    
    for i in range(num_images):
        # Take a screenshot
        screenshot = pyautogui.screenshot()
        screenshot.save(os.path.join(folder_path, f"{site_name}_{i + 1}.png"))
        print(f"Saved {site_name} screenshot {i + 1}")
        time.sleep(2)  # Adjust this wait time as needed between screenshots

# Collect screenshots for each site specified in the configuration
for site_name, site_info in data_collection_config['sites'].items():
    encoded_url = quote_plus(site_info['url'])
    take_screenshots(encoded_url, site_name, site_info['num_images'])
