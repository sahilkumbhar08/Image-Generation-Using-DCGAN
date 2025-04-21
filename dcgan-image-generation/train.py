import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from config import *
from models.generator import Generator
from models.discriminator import Discriminator
from utils.utils import save_generated_images

# Data loading
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = dsets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Generator(latent_dim).to(device)
D = Discriminator().to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizerG = optim.Adam(G.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizerD = optim.Adam(D.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# Training loop
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        # Train Discriminator
        D.zero_grad()
        real_images = real_images.to(device)
        b_size = real_images.size(0)
        labels_real = torch.full((b_size,), 1.0, dtype=torch.float, device=device)
        output_real = D(real_images).view(-1)
        lossD_real = criterion(output_real, labels_real)

        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        fake_images = G(noise)
        labels_fake = torch.full((b_size,), 0.0, dtype=torch.float, device=device)
        output_fake = D(fake_images.detach()).view(-1)
        lossD_fake = criterion(output_fake, labels_fake)
        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # Train Generator
        G.zero_grad()
        output_fake = D(fake_images).view(-1)
        lossG = criterion(output_fake, labels_real)
        lossG.backward()
        optimizerG.step()

    print(f"Epoch [{epoch+1}/{epochs}]  Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}")
    save_generated_images(fake_images[:64], epoch+1, output_path)