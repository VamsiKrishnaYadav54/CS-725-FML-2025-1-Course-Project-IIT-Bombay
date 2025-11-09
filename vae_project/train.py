# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Device setup for CPU/GPU portability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data loaders for MNIST
transform = transforms.Compose([transforms.ToTensor()])  # Normalize to [0,1]
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Encoder: Maps x (784D) to mu and logvar (20D latent)
class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten batch of images to [batch, 784]
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# Decoder: Maps z (20D) to logits (784D) for Bernoulli
class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        logits = self.fc_out(h)  # Logits for sigmoid/BCE
        return logits

# Full VAE model
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # sigma = exp(0.5 * log sigma^2)
        eps = torch.randn_like(std)    # Noise from standard normal
        z = mu + eps * std             # Reparameterized sample
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        return logits, mu, logvar

# Scaffold: Initialize model and test forward pass
if __name__ == "__main__":
    model = VAE().to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Test with a batch
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.to(device)
        logits, mu, logvar = model(x)
        print(f"Batch {batch_idx}: Input shape {x.shape}, Logits {logits.shape}, Mu {mu.shape}, Logvar {logvar.shape}")
        if batch_idx == 0:  # Just one batch for scaffold
            break
    print("Day 1 scaffold complete: Data and model forward pass verified.")

