import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import math


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


def load_checkpoint(checkpoint_path, device='cpu'):
    generator = UNetGenerator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return generator


def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Thêm batch dimension


def generate_sketches_from_checkpoints(checkpoints, image_path):
    input_image = load_image(image_path)

    num_checkpoints = len(checkpoints)
    num_images = num_checkpoints + 1  # Thêm ảnh gốc
    num_cols = 3  # Số cột trong lưới
    num_rows = math.ceil(num_images / num_cols)  # Số hàng cần để hiển thị

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()

    # Hiển thị ảnh gốc ở ô đầu tiên
    input_image_np = (input_image.squeeze(0).permute(1, 2, 0) * 0.5 + 0.5).cpu().numpy()
    axes[0].imshow(input_image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")


    # Hiển thị ảnh từ các checkpoint
    for i, checkpoint_path in enumerate(checkpoints):
        generator = load_checkpoint(checkpoint_path)

        with torch.no_grad():
            generated_sketch = generator(input_image)

        generated_sketch = (generated_sketch.squeeze(0).permute(1, 2, 0) * 0.5 + 0.5).cpu().numpy()

        axes[i + 1].imshow(generated_sketch)
        axes[i + 1].set_title(f"Checkpoint {checkpoint_path.split('_')[-1].split('.')[0]}")
        axes[i + 1].axis("off")

    # Ẩn các ô trống nếu không đủ ảnh để lấp đầy lưới
    for j in range(num_checkpoints + 1, len(axes)):
        axes[j].axis("off")

    plt.show()


if __name__ == "__main__":
    checkpoints = [
        "checkpoint2s/checkpoint_epoch_10.pth",
        "checkpoint2s/checkpoint_epoch_20.pth",
        "checkpoint2s/checkpoint_epoch_30.pth",
        "checkpoint2s/checkpoint_epoch_40.pth",
        "checkpoint2s/checkpoint_epoch_50.pth",
        "checkpoint2s/checkpoint_epoch_60.pth",
        "checkpoint2s/checkpoint_epoch_70.pth",
        "checkpoint2s/checkpoint_epoch_80.pth",
        "checkpoint2s/checkpoint_epoch_90.pth",
        "checkpoint2s/checkpoint_epoch_100.pth"
    ]
    test_image_path = "D:/Study/trainAI/dataset/label_test/car_01.jpg"

    generate_sketches_from_checkpoints(checkpoints, test_image_path)
