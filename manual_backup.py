
from helper import PatientDicomDataset

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
if not os.path.exists('data/Images/NSCLC Radiogenomics'):
         raise FileNotFoundError("The directory 'data/Images/NSCLC Radiogenomics' does not exist.")
if not os.path.exists('data/df_zscore.csv'):
         raise FileNotFoundError("The file 'data/df_zscore.csv' does not exist.")
# Verify that the dataset is properly loaded
dataset = PatientDicomDataset(root_dir='data/Images/NSCLC Radiogenomics', csv_path='data/df_zscore.csv', transform=transform)

# Check the length of the dataset to ensure it is not empty
if len(dataset) == 0:
    raise ValueError("The dataset is empty. Please check the root_dir and csv_path for correctness.")


images, patient_id, gene_values, gene_names = dataset[0]
print("Image tensor shape:", images[0].shape)

from torch.utils.data import random_split

# Define the split sizes
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Perform the split
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Test set size: {len(test_dataset)}")


from model import MedicalImageCNN, CNNToRNA, train_model
from torch.utils.data import DataLoader
import torch.nn as nn
from helper import collate_fn

# Create final model
model = CNNToRNA(
    cnn_encoder=MedicalImageCNN(output_dim=128),
    embedding_dim=128,
    output_dim=len(gene_names)
)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and loss
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training params
num_epochs = 10
batch_size = 32

# DataLoaders (make sure you use collate_fn if images are lists!)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

train_losses, val_losses = train_model(
    model, train_loader, val_loader,
    criterion, optimizer, device,
    num_epochs=3
)

#construct the graph so we can visualize the training and validation loss
import matplotlib.pyplot as plt
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
# Save the model (for walking away lmao)
torch.save(model.state_dict(), 'model.pth')

from sklearn.metrics import mean_squared_error, r2_score
from model import test_model

preds, labels = test_model(model, test_loader, device)

print("Test MSE:", mean_squared_error(labels, preds))
print("Test R² :", r2_score(labels, preds))  # Good for regression quality