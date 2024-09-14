import torch
from torchvision.utils import save_image
import os
from generator import Generator

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained generator model
generator = Generator().to(device)
generator.load_state_dict(torch.load('../models/generator_final.pth'))
generator.eval()  # Set the generator to evaluation mode

# Generate new images
num_images = 25
z = torch.randn(num_images, 100).to(device)
with torch.no_grad():  # No need to track gradients for inference
    generated_imgs = generator(z)

# Save generated images
if not os.path.exists('../generated_images'):
    os.makedirs('../generated_images')
save_image(generated_imgs.data, '../generated_images/generated.png', nrow=5, normalize=True)

print("Generated images saved to '../generated_images/generated.png'")