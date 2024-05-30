import os
import random
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms, models
from torchvision.io import read_image
from skimage import io

from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.gridspec as gridspec


class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for file_name in os.listdir(directory):
            if file_name.endswith('.jpg'):
                label = file_name.split('(')[0]
                self.image_paths.append(os.path.join(directory, file_name))
                self.labels.append(label)
        
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = read_image(image_path)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, image_path

weights=ResNet50_Weights.DEFAULT


model = models.resnet50(weights=weights)
model.eval()

transform = weights.transforms()

train_dataset = ImageDataset(directory='Train', transform=transform)
test_dataset = ImageDataset(directory='Test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_dataset.label_encoder.classes_))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

def train():

    num_epochs = 11

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            #transform here instead
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

    torch.save(model.state_dict(), 'model.pth')

def test_and_show_misclassifications(model, test_loader, label_encoder, device):
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            for i in range(images.size(0)):
                if predicted[i] != labels[i]:
                    misclassified_images.append(images[i].cpu())
                    misclassified_labels.append(labels[i].cpu())
                    misclassified_preds.append(predicted[i].cpu())

    num_misclassifications = len(misclassified_images)
    print(f'Number of misclassifications: {num_misclassifications}')

    # Display some misclassified images
    if num_misclassifications > 0:
        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(10):
            if i >= num_misclassifications:
                break
            ax = axs[i // 5, i % 5]
            img = F.to_pil_image(misclassified_images[i])
            true_label = label_encoder.inverse_transform([misclassified_labels[i].item()])[0]
            pred_label = label_encoder.inverse_transform([misclassified_preds[i].item()])[0]
            ax.imshow(img)
            ax.set_title(f'True: {true_label}\nPred: {pred_label}')
            ax.axis('off')
        plt.tight_layout()
        plt.show()


def show_random_predictions(model, test_dataset, label_encoder, device, num_images=3):
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    indices = random.sample(range(len(test_dataset)), num_images)
    
    fig = plt.figure(figsize=(15, 5 * num_images))
    gs = gridspec.GridSpec(num_images, 2, width_ratios=[1, 1.5])
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            _, label, image_path = test_dataset[idx]
            image_tensor = _.unsqueeze(0).to(device)  # Add batch dimension and move to device
            
            outputs = model(image_tensor)
            probabilities = nn.functional.softmax(outputs, dim=1)
            probabilities = probabilities.cpu().numpy()[0]  # Move to CPU and get the first item
            
            sorted_indices = np.argsort(-probabilities)  # Sort in descending order
            sorted_probabilities = probabilities[sorted_indices]
            sorted_labels = label_encoder.inverse_transform(sorted_indices)
            
            # Show original image
            ax_img = plt.subplot(gs[i, 0])
            original_image = io.imread(image_path)
            ax_img.imshow(original_image)
            ax_img.axis('off')
            
            true_label = label_encoder.inverse_transform([label])[0]
            ax_img.set_title(f"True: {true_label}")

            # Show top predictions as bar graph
            ax_bar = plt.subplot(gs[i, 1])
            top_predictions = {sorted_labels[j]: sorted_probabilities[j] for j in range(5)}
            
            labels = list(top_predictions.keys())
            probs = list(top_predictions.values())
            ax_bar.barh(labels, probs)
            ax_bar.set_xlim(0, 1)
            ax_bar.set_xlabel('Probability')
            ax_bar.invert_yaxis()  # To have the highest probability on top

    plt.tight_layout()
    plt.show()


def main():
    #train()
    #test_and_show_misclassifications(model, test_loader, train_dataset.label_encoder, device)
    show_random_predictions(model, test_dataset, train_dataset.label_encoder, device, num_images=3)


main()