import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from unet_model import UNet  # Import U-Net model
from dataset import OralHealthDataset, transform  # Import dataset class & transformations

# ✅ Initialize dataset & dataloader with 512x512 resolution
train_dataset = OralHealthDataset("data/images", "data/masks", transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# ✅ Initialize the model (5 classes including background)
model = UNet(in_channels=3, out_channels=5)

# ✅ Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Define the loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ✅ Training loop
num_epochs = 50

print(f"Starting training on {device} with 512x512 images...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# ✅ Save trained model
torch.save(model.state_dict(), "unet_model_512.pth")
print("Model training complete & saved as unet_model_512.pth")
