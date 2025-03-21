import os
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm

# Dataset class
class CarToSketchDataset(Dataset):
    def __init__(self, car_dir, label_dir, transform=None):
        self.car_files = sorted(os.listdir(car_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.car_dir = car_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.car_files)

    def __getitem__(self, idx):
        car_path = os.path.join(self.car_dir, self.car_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        car_image = Image.open(car_path).convert("RGB")
        label_image = Image.open(label_path).convert("RGB")

        if self.transform:
            car_image = self.transform(car_image)
            label_image = self.transform(label_image)

        return car_image, label_image


# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
dataset = CarToSketchDataset(car_dir="D:/Study/trainAI/dataset/label", label_dir="D:/Study/trainAI/dataset/label_test",
                             transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# Define UNet Generator
class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()

        def down_block(in_channels, out_channels, apply_batchnorm=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
            if apply_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up_block(in_channels, out_channels, dropout=False):
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
            layers.append(nn.BatchNorm2d(out_channels))
            if dropout:
                layers.append(nn.Dropout(0.5))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.down1 = down_block(3, 64, apply_batchnorm=False)
        self.down2 = down_block(64, 128)
        self.down3 = down_block(128, 256)
        self.down4 = down_block(256, 512)
        self.down5 = down_block(512, 512)

        self.up1 = up_block(512, 512, dropout=True)
        self.up2 = up_block(1024, 256)
        self.up3 = up_block(512, 128)
        self.up4 = up_block(256, 64)
        self.final = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5)
        u2 = self.up2(torch.cat([u1, d4], dim=1))
        u3 = self.up3(torch.cat([u2, d3], dim=1))
        u4 = self.up4(torch.cat([u3, d2], dim=1))
        out = self.final(torch.cat([u4, d1], dim=1))
        return self.tanh(out)


# Define PatchGAN Discriminator
class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()

        def block(in_channels, out_channels, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            block(6, 64, stride=2),
            block(64, 128, stride=2),
            block(128, 256, stride=2),
            block(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))


# Initialize models and optimizers
generator = UNetGenerator()
discriminator = PatchGANDiscriminator()
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# find lastest checkpoint
def find_latest_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    if not checkpoints:
        return None, 0
    checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]), reverse=True)
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[0])
    latest_epoch = int(checkpoints[0].split("_")[-1].split(".")[0])
    return latest_checkpoint, latest_epoch


# check folder checkpoints
checkpoint_dir = "checkpoint1s"
os.makedirs(checkpoint_dir, exist_ok=True)

# Load checkpoint if had
latest_checkpoint, epoch_start = find_latest_checkpoint(checkpoint_dir)

generator = UNetGenerator()
discriminator = PatchGANDiscriminator()

criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

if latest_checkpoint:
    checkpoint = torch.load(latest_checkpoint)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
    print(f"Loaded checkpoint from {latest_checkpoint}, resuming from epoch {epoch_start}")
else:
    print("No checkpoint found, starting from scratch.")

# training
total_epochs = 100
for epoch in range(epoch_start, total_epochs):
    loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)

    for i, (real_car, sketch) in loop:
        fake_sketch = generator(real_car)

        loss_G = criterion_GAN(discriminator(fake_sketch, real_car),
                               torch.ones_like(discriminator(fake_sketch, real_car))) + criterion_L1(fake_sketch,
                                                                                                     sketch) * 100
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        loss_D = (criterion_GAN(discriminator(sketch, real_car), torch.ones_like(discriminator(sketch, real_car))) +
                  criterion_GAN(discriminator(fake_sketch.detach(), real_car),
                                torch.zeros_like(discriminator(fake_sketch.detach(), real_car)))) / 2
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # update terminal
        loop.set_description(f"Epoch [{epoch + 1}/{total_epochs}]")
        loop.set_postfix(Loss_G=loss_G.item(), Loss_D=loss_D.item())

    print(f"Epoch [{epoch + 1}/{total_epochs}] - Loss_G: {loss_G.item():.4f} - Loss_D: {loss_D.item():.4f}")

    # save checkpoint
    if (epoch + 1) % 10 == 0:
        checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict()
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
