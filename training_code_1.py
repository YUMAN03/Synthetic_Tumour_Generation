"""
TUMOR INPAINTING GAN PIPELINE - PART 2: TRAINING
1. Creates dataset from preprocessed patches
2. Defines GAN architecture
3. Trains the model
4. Saves generator checkpoint
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from glob import glob

# Reuse paths from preprocessing
OUTPUT_DIR_MASKED = "/home/shivam/tumor_deepfake_GAN/masked_extended_tumors"
OUTPUT_DIR_EXTENDED = "/home/shivam/tumor_deepfake_GAN/extended_tumors"

class TumorDataset(Dataset):
    """Paired dataset of masked patches and real tumor patches"""
    def __init__(self, masked_dir, real_dir, transform=None):
        self.masked_dir = masked_dir
        self.real_dir = real_dir
        self.transform = transform
        self.filenames = [f for f in os.listdir(masked_dir) if f.endswith('_masked.png')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        masked_path = os.path.join(self.masked_dir, self.filenames[idx])
        real_path = os.path.join(self.real_dir, self.filenames[idx].replace("_masked", ""))
        
        masked_img = Image.open(masked_path).convert("L")
        real_img = Image.open(real_path).convert("L")

        if self.transform:
            masked_img = self.transform(masked_img)
            real_img = self.transform(real_img)

        return masked_img, real_img

class Generator(nn.Module):
    """U-Net like architecture for tumor generation"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class Discriminator(nn.Module):
    """PatchGAN discriminator"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256*8*8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_gan():
    """Complete GAN training pipeline"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Prepare data
    dataset = TumorDataset(OUTPUT_DIR_MASKED, OUTPUT_DIR_EXTENDED, transform)
    train_size = int(0.8 * len(dataset))
    train_set, _ = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Loss and optimizers
    adv_loss = nn.BCELoss()
    recon_loss = nn.L1Loss()
    opt_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(100):  # <-- This should be indented under train_gan()
        generator.train()
        discriminator.train()
        
        # Initialize epoch accumulators
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0
        
        for masked, real in train_loader:
            masked, real = masked.to(device), real.to(device)
            batch_size = masked.size(0)
            
            # --- Discriminator Training ---
            opt_d.zero_grad()
            real_loss = adv_loss(discriminator(real), torch.ones(batch_size, 1, device=device))
            fake_loss = adv_loss(discriminator(generator(masked).detach()), torch.zeros(batch_size, 1, device=device))
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_d.step()
            
            # --- Generator Training ---
            opt_g.zero_grad()
            gen_imgs = generator(masked)
            g_adv = adv_loss(discriminator(gen_imgs), torch.ones(batch_size, 1, device=device))
            g_recon = recon_loss(gen_imgs, real)
            g_loss = g_adv + 100 * g_recon
            g_loss.backward()
            opt_g.step()
            
            # Accumulate losses
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            num_batches += 1
        
        # Print average losses per epoch
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        print(f"Epoch {epoch+1}, Avg D Loss: {avg_d_loss:.4f}, Avg G Loss: {avg_g_loss:.4f}")

    # Save model (indented under train_gan())
    torch.save(generator.state_dict(), "/home/shivam/tumor_deepfake_GAN/generator.pth")
    print("Training completed and generator saved")

if __name__ == "__main__":
    train_gan()
