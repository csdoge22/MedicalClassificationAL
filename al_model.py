from datasets import load_dataset, DatasetDict
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------
# 🟢 Load and split the dataset
# ---------------------------------------------------
dataset = load_dataset(path="shrashraddha/medical_image_cleaning", cache_dir="./cache", split="train")

# Large dataset - split carefully, no batch=True mapping on entire dataset to save memory/time
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_val = split_dataset['train']
test = split_dataset['test']
split_train_val = train_val.train_test_split(test_size=0.1, seed=42)
train = split_train_val['train']
val = split_train_val['test']

dataset = DatasetDict({
    "train": train,
    "validation": val,
    "test": test
})

# ---------------------------------------------------
# 🟨 Encode multi-word class labels to integers
# ---------------------------------------------------
unique_labels = sorted(set(dataset["train"]["txt"]))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
num_classes = len(label2id)

def encode_label(example):
    example["label"] = label2id[example["txt"]]
    return example

for split in ["train", "validation", "test"]:
    dataset[split] = dataset[split].map(encode_label)

# ---------------------------------------------------
# 🎨 Image transforms (convert grayscale to RGB)
# ---------------------------------------------------
def to_rgb(image):
    return image.convert("RGB")

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(to_rgb),  # convert grayscale to RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    "validation": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(to_rgb),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(to_rgb),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
}

# ---------------------------------------------------
# Custom Dataset wrapper to apply transforms on the fly
# ---------------------------------------------------
from torch.utils.data import Dataset

class HuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        image = self.transform(image)
        return image, label

# Wrap datasets to apply transforms on-the-fly instead of map (memory efficient for large datasets)
train_dataset = HuggingFaceDataset(dataset["train"], data_transforms["train"])
val_dataset = HuggingFaceDataset(dataset["validation"], data_transforms["validation"])
test_dataset = HuggingFaceDataset(dataset["test"], data_transforms["test"])

# ---------------------------------------------------
# 📦 DataLoaders with multiple workers for speed
# ---------------------------------------------------
batch_size = 16  # increase batch size from 4 for better throughput if memory allows
num_workers = 0  # adjust based on your CPU cores

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# ---------------------------------------------------
# 🧠 CNN Model (no changes)
# ---------------------------------------------------
class MedicalCNN(nn.Module):
    def __init__(self, num_classes):
        super(MedicalCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 56 * 56, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------------------------------
# 🏋️ Training Setup (no changes)
# ---------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MedicalCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# ---------------------------------------------------
# 🔁 Training and Validation Loop (no changes)
# ---------------------------------------------------
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)

    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels)

    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    val_accuracy = val_correct.double() / len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Val Accuracy: {val_accuracy:.4f}")
