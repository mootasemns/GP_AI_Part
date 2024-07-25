import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
import json


# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)
    
    
# Load the dataset
df = pd.read_csv(config['data_path'])

df.dropna(inplace=True)

# Split data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.1)
 
# Convert dataframes into Hugging Face dataset objects
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)


