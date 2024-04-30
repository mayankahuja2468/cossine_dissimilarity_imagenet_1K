import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
import os

def save_pretrained_resnet50(model_directory):
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    os.makedirs(model_directory, exist_ok=True)
    model_path = os.path.join(model_directory, 'resnet50_imagenet_v2.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    script_directory = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_directory)
    model_directory = os.path.join(project_root, 'Models')
    save_pretrained_resnet50(model_directory)
