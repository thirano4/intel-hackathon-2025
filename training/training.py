import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
import os

device = 'xpu' if torch.xpu.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

img_dir = os.path.join("training", "JPEGImages")
label_file = 'training\labels.xlsx'

df_raw = pd.read_excel(label_file)
df_raw.columns = df_raw.columns.map(lambda x: str(x).strip())

label_col = df_raw.columns[1] 

df = pd.DataFrame({
    'filename': df_raw.iloc[:, 0].apply(lambda x: f"BloodImage_{int(x):05d}.jpg"),
    'label': df_raw[label_col].astype(str).str.strip().str.upper()
})

df['exists'] = df['filename'].apply(lambda f: os.path.exists(os.path.join(img_dir, f)))
df = df[df['label'].map(df['label'].value_counts()) >= 2]

def file_exists(filename):
    path = os.path.join(img_dir, filename)
    return os.path.isfile(path)

df = df[df['filename'].apply(file_exists)].reset_index(drop=True)

def __getitem__(self, idx):
    row = self.df.iloc[idx]
    img_path = os.path.join(self.img_dir, row['filename'])

    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = Image.open(img_path).convert('RGB')
    label = class_to_idx[row['label']]
    if self.transform:
        image = self.transform(image)
    return image, label

class BloodCellDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = class_to_idx[row['label']]
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_dataset = BloodCellDataset(train_df, img_dir, transform=transform)
test_dataset = BloodCellDataset(test_df, img_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

classes = sorted(df['label'].unique())
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model.train()
use_autocast = device == "cuda"

for epoch in range(1):
    running_loss = 0.0
    print(f"Starting Epoch {epoch + 1}")
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        if use_autocast:
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
    print(f"Epoch {epoch + 1} completed. Avg Loss: {running_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

torch.save(model.state_dict(), "bloodcell_resnet18.pth")
print(f"Accuracy on test set: {100 * correct / total:.2f}%")
