import os
os.environ["ZE_ENABLE_TRACING_LAYER"] = "1"

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import requests
from torchvision.models.resnet import ResNet18_Weights
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

#image manipulation library
from PIL import Image
from PIL import ImageDraw

device = torch.device('xpu' if torch.xpu.is_available() else 'cpu')

# Create a tensor on the XPU device
tensor = torch.ones(3, 4, device=device)

# Matrix multiplication
mat1 = torch.randn(3, 4, device=device)
mat2 = torch.randn(4, 5, device=device)
result = torch.matmul(mat1, mat2)

# Get model

classes = [
  "BASOPHIL",
  "EOSINOPHIL",
  "NEUTROPHIL",
  "LYMPHOCYTE",
  "MONOCYTE",
  "NEUTROPHIL",
  "OSINOPHIL",
  "NEUTROPHIL, EOSINOPHIL",
  "NEUTROPHIL,BASOPHIL"
]


#classes = ["","BASOPHIL","EOSINOPHIL","EUTROPHIL","LYMPHOCYTE","MONOCYTE","NEUTROPHIL","NEUTROPHIL, EOSINOPHIL","NEUTROPHIL", "NEUTROPHIL"]

def identify(image):
    model = models.resnet18(weights=None)  # No pretrained weights needed here, since we load state_dict
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load("training/Model/bloodcell_resnet18.pth", map_location=device))
    model.to(device)
    model.eval()

    # Prepare the input image 
    input_image = Image.open(image)
    #input_image = Image.open(requests.get(image_url, stream=True).raw)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # infer
    model = model.eval()
    with torch.no_grad():
        output = model(input_batch)

    _, predicted = torch.max(output, 1)
    class_index = predicted.item()
    class_label = classes[class_index]

    print(f"Predicted class: {class_index}")
    print(f"Class label: {class_label}")
    return [class_index, class_label]
