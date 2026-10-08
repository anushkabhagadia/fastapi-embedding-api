"""Train the CNN on CIFAR10 (images resized to 64x64) and save weights."""
import torch, torch.nn as nn, torch.optim as optim
import torchvision, torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from cnn_model import CNN

EPOCHS, BATCH_SIZE = 10, 32
torch.manual_seed(42)
device = (torch.device("mps") if torch.backends.mps.is_available()
          else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

# CIFAR10 is 32x32, the spec says 64x64 -> resize
transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
train_ds = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
test_ds = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=100):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}: avg loss {running_loss/len(train_loader):.4f}")

model.eval()
correct = total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
print(f"Test accuracy: {100*correct/total:.2f}%")

torch.save(model.state_dict(), "cnn_cifar10.pt")   # commit this file (~3 MB) so Docker can load it
