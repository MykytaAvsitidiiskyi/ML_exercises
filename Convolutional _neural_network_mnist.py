import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # BatchNorm
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)  #
        self.bn2 = nn.BatchNorm2d(32)  # BatchNorm
        self.conv3 = nn.Conv2d(32, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # BatchNorm
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tensor_shapes_summary(model, input_tensor):
    def forward_hook(module, input, output):
        num_params = sum(p.numel() for p in module.parameters())
        print(f"{module.__class__.__name__.ljust(20)} | "
              f"Input shape: {str(input[0].shape).ljust(30)} | "
              f"Output shape: {str(output.shape).ljust(30)} | "
              f"Trainable Params: {num_params}")

    hooks = []
    for layer in model.children():
        hook = layer.register_forward_hook(forward_hook)
        hooks.append(hook)

    with torch.no_grad():
        model(input_tensor)

    for hook in hooks:
        hook.remove()


# Ф-ція для оверфіьа
def overfit_model(model, device, train_loader, test_loader, optimizer, loss_fn, epochs):
    train_losses = []
    test_losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fn(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_losses.append(test_loss / len(test_loader))

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_losses[-1]:.6f} | Test Loss: {test_losses[-1]:.6f} | Accuracy: {100.0 * correct / len(test_loader.dataset):.2f}%")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs + 1), test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.show()


#  Класіфікейшн репорт і  confusion matrix
def evaluate_model(model, device, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    print("Classification Report:")
    print(classification_report(all_targets, all_preds))

    conf_matrix = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:")
    print(conf_matrix)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def main():
    torch.manual_seed(1337)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_kwargs = {'batch_size': 128}
    test_kwargs = {'batch_size': 128}

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST('./mnsit-dataset', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./mnsit-dataset', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = Net().to(device)
    print("Number of Trainable Parameters:")
    print(count_parameters(model))

    input_tensor = torch.randn(1, 1, 28, 28).to(device)
    print("Tensor Shapes Before and After Each Layer:")
    tensor_shapes_summary(model, input_tensor)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    epochs = 3
    overfit_model(model, device, train_loader, test_loader, optimizer, loss_fn, epochs)


    model.conv2 = nn.Conv2d(32, 32, 3, padding=1)

    print("Reduced Number of Trainable Parameters:")
    print(count_parameters(model))

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    epochs = 3
    overfit_model(model, device, train_loader, test_loader, optimizer, loss_fn, epochs)


    evaluate_model(model, device, test_loader)


if __name__ == '__main__':
    main()
