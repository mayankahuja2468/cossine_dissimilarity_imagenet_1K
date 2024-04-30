import torch
import tarfile
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
from collections import defaultdict
from torchvision import models
from imagenet_classes import IMAGENET2012_CLASSES
import os

def get_class_embeddings_train():
    class_embeddings = defaultdict(list)
    base_path = '../Data'
    archive_patterns = [('train_images_{}.tar.gz', 5)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model_path = '../Models/resnet50_imagenet_v2.pth'
    model = models.resnet50()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

# Now you can use model to make predictions
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])    
    for pattern, count in archive_patterns:
        for i in range(count):
            file_path = os.path.join(base_path, pattern.format(i))          
            with tarfile.open(file_path, 'r:gz') as tar:
                for member in tar.getmembers():
                    if member.isfile() and member.name.endswith('.JPEG'):
                        synset_id = os.path.basename(member.name).split('_')[0]
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
    averaged_class_embeddings = {label: sum(embeds)/len(embeds) for label, embeds in class_embeddings.items()}
    return averaged_class_embeddings
