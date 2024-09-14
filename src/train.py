import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image
import os
from generator import Generator
from discriminator import Discriminator
from utils import get_dataloader

if torch.cuda.is_available():
    print("CUDA is available. You have a GPU!")
    device = torch.device("cuda")
else:
    print("CUDA is not available. You are using the CPU.")
    device = torch.device("cpu")

# Initialize Networks
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

# Loss Function
criterion = nn.BCELoss()

# Load Dataset
dataloader = get_dataloader()

# Training Loop
num_epochs = 50
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # Train Discriminator
        real_imgs = imgs.to(device)
        real_labels = torch.ones(imgs.size(0), 1).to(device)
        fake_labels = torch.zeros(imgs.size(0), 1).to(device)

        optimizer_D.zero_grad()
        outputs = discriminator(real_imgs)
        d_loss_real = criterion(outputs, real_labels)
        z = torch.randn(imgs.size(0), 100).to(device)
        fake_imgs = generator(z)
        outputs = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn(imgs.size(0), 100).to(device)
        fake_imgs = generator(z)
        outputs = discriminator(fake_imgs)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss D: {d_loss.item()}, Loss G: {g_loss.item()}")

    # Save generated images
    if not os.path.exists('../images'):
        os.makedirs('../images')
    
    # Debugging: Print the shape of the generated images
    print(f"Generated images shape: {fake_imgs.shape}")
    
    # Save the generated images
    image_path = f'../images/{epoch+1}.png'
    save_image(fake_imgs.data[:25], image_path, nrow=5, normalize=True)
    
    # Print the full path of the saved image
    full_image_path = os.path.abspath(image_path)
    print(f"Saved generated images for epoch {epoch+1} at {full_image_path}")

    # Save the model checkpoints
    if not os.path.exists('../models'):
        os.makedirs('../models')
    torch.save(generator.state_dict(), f'../models/generator_epoch_{epoch+1}.pth')
    torch.save(discriminator.state_dict(), f'../models/discriminator_epoch_{epoch+1}.pth')
    print(f"Saved model checkpoints for epoch {epoch+1}")

# Final model save
torch.save(generator.state_dict(), '../models/generator_final.pth')
torch.save(discriminator.state_dict(), '../models/discriminator_final.pth')
print("Saved final model checkpoints")