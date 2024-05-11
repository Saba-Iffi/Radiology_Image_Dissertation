import torch  
import torchvision.transforms as transforms 
from torchvision import datasets, transforms    
from torch.utils.data import DataLoader, random_split  
from torchvision.models import resnet18 
import torch.nn as nn   
import torch.optim as optim 
from sklearn.metrics import confusion_matrix, roc_curve, auc   
import matplotlib.pyplot as plt 
import numpy as np  

# Setup the computation device(Compute Unified Device Architecture)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations for normalization and resizing
transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#mounted drive from google drive
from google.colab import drive
from google.colab import files
drive.mount('/content/drive')

# Load images and create dataset
data_dir = '/content/drive/My Drive/dataset/dataset'  # Path to the dataset
full_dataset = datasets.ImageFolder(root=data_dir, transform=transformations)

# Split the dataset into 80% training and 20% testing
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# DataLoader setup for batch processing
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Model setup using pre-trained ResNet18
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model.to(device)

# Loss function and optimizer setup
criterion = nn.CrossEntropyLoss()   
optimizer = optim.Adam(model.parameters(), lr=0.001)   

# Model training with fix loops
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs} complete.')

# Save the trained model
torch.save(model.state_dict(), '/content/model.pth')
print("Model saved successfully.")

# Load the model for evaluation
model.load_state_dict(torch.load('/content/model.pth'))
model.eval()

# Model evaluation using test data
y_pred = []
y_true = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# Calculate and print confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Calculate ROC(Receiver Operating Characteristic) curve and AUC(Area Under the Curve)
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
print(roc_auc)  #how well the model is performing overall------>{higher AUC value (closer to 1) indicates better performance, while a lower value (closer to 0.5) suggests poorer performance
print(tpr)  #true positive rates
print(fpr)  #false positive rates
print(thresholds)

# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

# Calculate precision
precision = precision_score(y_true, y_pred)

# Calculate recall (sensitivity)
recall = recall_score(y_true, y_pred)

# Calculate specificity
conf_matrix = confusion_matrix(y_true, y_pred)
true_negatives = conf_matrix[0, 0]
false_positives = conf_matrix[0, 1]
specificity = true_negatives / (true_negatives + false_positives)

# Calculate F1 score
f1_score = 2 * (precision * recall) / (precision + recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall (Sensitivity):", recall)
print("Specificity:", specificity)
print("F1 Score:", f1_score)