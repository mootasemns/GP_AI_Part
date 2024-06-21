
from torch import nn 
import torchvision 
from utils import readConfig

model_config = readConfig('config/model_config.yaml')

class CheatDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(CheatDetectionModel, self).__init__()
        model_name = model_config['MODEL']['name']
        
        # Load the model with the specified weights
        self.base_model = torchvision.models.__dict__[model_name](
            weights=torchvision.models.get_model_weights(model_name).DEFAULT
        )
        
        # Replace the final layer with a custom layer for our specific number of classes
        if model_name.startswith('resnet'):
            num_ftrs = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name.startswith('vgg') or model_name.startswith('alexnet'):
            num_ftrs = self.base_model.classifier[6].in_features
            self.base_model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        else:
            raise NotImplementedError(f'Model {model_name} not supported.')

    def forward(self, x):
        return self.base_model(x)