
"""
GAN-based Data Augmentation for Plant Disease Images
---------------------------------------------------
This script implements a simple fully-connected GAN (Generator + Discriminator)
to synthesize plant disease images for data augmentation purposes.

Author: Sonu Varghese K
License: MIT
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.utils import save_image
import os


# ---------------- Configuration ----------------
IMAGE_SIZE = 256
LATENT_DIM = 100
BATCH_SIZE = 128
EPOCHS = 5000
DATASET_PATH = "data/original_dataset"   # Update path
OUTPUT_DIR = "generated_images"
LR = 0.0002
BETAS = (0.5, 0.999)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------- Dataset ----------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = torchvision.datasets.ImageFolder(
    root=DATASET_PATH,
    transform=transform
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)


# ---------------- Models ----------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128 * 128 * 3),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.model(z)
        return out.view(-1, 3, 128, 128)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(128 * 128 * 3, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)


generator = Generator().to(device)
discriminator = Discriminator().to(device)


# ---------------- Training Setup ----------------
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=BETAS)
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=BETAS)


# ---------------- Training Loop ----------------
for epoch in range(EPOCHS):
    for real_images, _ in dataloader:
        real_images = real_images.to(device, non_blocking=True)
        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # Train Discriminator
        optimizer_D.zero_grad(set_to_none=True)

        real_loss = criterion(discriminator(real_images), real_labels)

        z = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_images = generator(z)
        fake_loss = criterion(discriminator(fake_images.detach()), fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad(set_to_none=True)
        g_loss = criterion(discriminator(fake_images), real_labels)
        g_loss.backward()
        optimizer_G.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")
        resized = F.interpolate(fake_images, size=(IMAGE_SIZE, IMAGE_SIZE),
                                mode='bilinear', align_corners=False)
        save_image(resized[0], f"{OUTPUT_DIR}/epoch_{epoch+1}.png", normalize=True)

print("Training complete. Synthetic images saved to:", OUTPUT_DIR)
