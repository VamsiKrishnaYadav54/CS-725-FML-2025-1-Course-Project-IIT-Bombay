"""
Variational Autoencoder reproduction (Kingma & Welling, 2013) — PyTorch
Trains a small VAE on MNIST.

Requirements:
pip install torch torchvision matplotlib

Save as vae_mnist.py and run: python vae_mnist.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
import random
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Reproducibility helpers
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ---------------------------
# Hyperparameters
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 20
learning_rate = 1e-3
latent_dim = 20  # z dimension (Kingma & Welling often used 20)
hidden_dim = 400  # hidden units in encoder/decoder
log_interval = 200
save_dir = "vae_results"
os.makedirs(save_dir, exist_ok=True)

# ---------------------------
# Data (MNIST)
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),  # pixels in [0,1]
])

train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# ---------------------------
# VAE model
# ---------------------------
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mu
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # logvar

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        # Initialization (optional)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick: z = mu + sigma * eps, eps ~ N(0,1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # Bernoulli decoder (outputs logits / probabilities in [0,1])
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# ---------------------------
# Loss (ELBO)
# ---------------------------
def loss_function(recon_x, x, mu, logvar):
    """
    For binary MNIST we use Bernoulli likelihood -> reconstruction loss is
    negative log-likelihood = binary cross-entropy (elementwise).
    KL divergence between q(z|x)=N(mu, sigma^2) and p(z)=N(0,I) has closed form:
    KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    We return total loss = BCE + KL (averaged over batch).
    """
    # Reconstruction loss (binary cross entropy)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')  # sum over elements in batch
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD

# ---------------------------
# Training / Testing routines
# ---------------------------
model = VAE(input_dim=784, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train(epoch):
    model.train()
    train_loss = 0.0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss, bce, kld = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                  f"Loss: {loss.item() / len(data):.4f} (BCE {bce.item()/len(data):.4f} | KLD {kld.item()/len(data):.4f})")

    avg_loss = train_loss / len(train_loader.dataset)
    print(f"====> Epoch: {epoch} Average loss: {avg_loss:.4f}")
    return avg_loss

def test(epoch):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon, mu, logvar = model(data)
            loss, bce, kld = loss_function(recon, data, mu, logvar)
            test_loss += loss.item()

            if i == 0:
                # save first batch reconstructions (input, reconstruction)
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon.view(-1, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), os.path.join(save_dir, f'reconstruction_{epoch}.png'), nrow=n)

    test_loss /= len(test_loader.dataset)
    print(f"====> Test set loss: {test_loss:.4f}")
    return test_loss

# ---------------------------
# Sampling / Latent traversals
# ---------------------------
@torch.no_grad()
def sample_and_save(epoch, n=64):
    model.eval()
    z = torch.randn(n, latent_dim, device=device)
    samples = model.decode(z).cpu().view(-1, 1, 28, 28)
    save_image(samples, os.path.join(save_dir, f'samples_{epoch}.png'), nrow=8)

@torch.no_grad()
def latent_traversal(save_path_prefix="traversal", steps=11):
    """
    For the first two latent dims, vary them from -3 to +3 and keep others 0.
    Save a grid showing how reconstruction changes.
    """
    model.eval()
    zs = []
    rng = torch.linspace(-3, 3, steps)
    for a in rng:
        for b in rng:
            z = torch.zeros(latent_dim, device=device)
            z[0] = a
            z[1] = b
            zs.append(z)
    zs = torch.stack(zs, dim=0)
    recon = model.decode(zs).cpu().view(-1, 1, 28, 28)
    save_image(recon, f"{save_path_prefix}.png", nrow=steps)

# ---------------------------
# Run training
# ---------------------------
train_losses = []
test_losses = []
for epoch in range(1, epochs + 1):
    train_losses.append(train(epoch))
    test_losses.append(test(epoch))
    sample_and_save(epoch, n=64)

# Save model
torch.save(model.state_dict(), os.path.join(save_dir, "vae_mnist.pth"))

# Plot losses
plt.figure(figsize=(6, 4))
plt.plot(range(1, epochs + 1), train_losses, label="train_loss")
plt.plot(range(1, epochs + 1), test_losses, label="test_loss")
plt.xlabel("Epoch")
plt.ylabel("Average loss (per image)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "loss_curve.png"))
print("Finished. Results saved in", save_dir)
