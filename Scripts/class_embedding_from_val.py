import torch
import tarfile
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from collections import defaultdict
from torchvision import models
from imagenet_classes import IMAGENET2012_CLASSES
import os

def get_class_embeddings_val():
    class_embeddings = defaultdict(list)
    base_path = '../Data'
    archive_pattern = 'val_images.tar.gz'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Define the path to the model file
    model_path = '../Models/resnet50_imagenet_v2.pth'
    # Load the pretrained ResNet-50 model
    model = models.resnet50()
    # Load the weights from the .pth file
    model.load_state_dict(torch.load(model_path, map_location=device))
    # If you want to use the model for inference, set it to evaluation mode
    model.to(device)
    model.eval()

    # Preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])    

    # Open the tar file
    file_path = os.path.join(base_path, archive_pattern)          
    with tarfile.open(file_path, 'r:gz') as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith('.JPEG'):
                synset_id = member.name.split('_')[-1].split('.')[0]
                label = IMAGENET2012_CLASSES.get(synset_id)
                if label:                           
                    file = tar.extractfile(member)
                    img = Image.open(BytesIO(file.read()))
                    img = img.convert('RGB')
                    img_tensor = preprocess(img).unsqueeze(0).to(device)
                    with torch.no_grad():  
                        embedding = model(img_tensor)
                        embedding = embedding.cpu()
                    class_embeddings[label].append(embedding.squeeze().numpy())

    # Average the embeddings for each class
    averaged_class_embeddings = {label: sum(embeds)/len(embeds) for label, embeds in class_embeddings.items()}
    return averaged_class_embeddings
