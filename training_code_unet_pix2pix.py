"""
TUMOR INPAINTING GAN PIPELINE - PART 2: TRAINING (Pix2Pix + U-Net with Skip Connections)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# Directories
OUTPUT_DIR_MASKED = "/home/shivam/tumor_deepfake_GAN/masked_extended_tumors"
OUTPUT_DIR_EXTENDED = "/home/shivam/tumor_deepfake_GAN/extended_tumors"

# -----------------------------
# Dataset Definition
# -----------------------------
class TumorDataset(Dataset):
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

# -----------------------------
# Generator with U-Net + skip connections
# -----------------------------
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = self.conv_block(1, 64, norm=False)   # -> 64x32x32
        self.enc2 = self.conv_block(64, 128)             # -> 128x16x16
        self.enc3 = self.conv_block(128, 256)            # -> 256x8x8
        self.enc4 = self.conv_block(256, 512)            # -> 512x4x4

        # Decoder
        self.dec1 = self.deconv_block(512, 256)          # -> 256x8x8
        self.dec2 = self.deconv_block(512, 128)          # -> 128x16x16
        self.dec3 = self.deconv_block(256, 64)           # -> 64x32x32
        self.dec4 = nn.Sequential(                       # -> 1x64x64
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh()
        )

    def conv_block(self, in_c, out_c, norm=True):
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
        if norm:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def deconv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)   # 64x32x32
        e2 = self.enc2(e1)  # 128x16x16
        e3 = self.enc3(e2)  # 256x8x8
        e4 = self.enc4(e3)  # 512x4x4

        # Decoder with skip connections
        d1 = self.dec1(e4)              # 256x8x8
        d2 = self.dec2(torch.cat([d1, e3], dim=1))  # 128x16x16
        d3 = self.dec3(torch.cat([d2, e2], dim=1))  # 64x32x32
        out = self.dec4(torch.cat([d3, e1], dim=1)) # 1x64x64
        return out

# -----------------------------
# PatchGAN Discriminator
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, masked, target):
        x = torch.cat([masked, target], dim=1)
        return self.model(x)

# -----------------------------
# Training Function
# -----------------------------
def train_gan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = TumorDataset(OUTPUT_DIR_MASKED, OUTPUT_DIR_EXTENDED, transform)
    train_size = int(0.8 * len(dataset))
    train_set, _ = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

    generator = UNetGenerator().to(device)
    discriminator = Discriminator().to(device)

    adv_loss = nn.BCELoss()
    recon_loss = nn.L1Loss()

    opt_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(500):
        generator.train()
        discriminator.train()
        d_total, g_total, batches = 0, 0, 0

        for masked, real in train_loader:
            masked, real = masked.to(device), real.to(device)
            batch_size = masked.size(0)

            # Train Discriminator
            opt_d.zero_grad()
            real_pred = discriminator(masked, real)
            real_loss = adv_loss(real_pred, torch.ones_like(real_pred))

            fake_imgs = generator(masked).detach()
            fake_pred = discriminator(masked, fake_imgs)
            fake_loss = adv_loss(fake_pred, torch.zeros_like(fake_pred))

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_d.step()

            # Train Generator
            opt_g.zero_grad()
            gen_imgs = generator(masked)
            gen_pred = discriminator(masked, gen_imgs)

            g_adv = adv_loss(gen_pred, torch.ones_like(gen_pred))
            g_l1 = recon_loss(gen_imgs, real)
            g_loss = g_adv + 100 * g_l1
            g_loss.backward()
            opt_g.step()

            d_total += d_loss.item()
            g_total += g_loss.item()
            batches += 1

        print(f"Epoch {epoch+1}, D Loss: {d_total / batches:.4f}, G Loss: {g_total / batches:.4f}")

    torch.save(generator.state_dict(), "/home/shivam/tumor_deepfake_GAN/generator_pix2pix_unet.pth")
    print("Training complete. Generator with U-Net saved.")

if __name__ == "__main__":
    train_gan()

