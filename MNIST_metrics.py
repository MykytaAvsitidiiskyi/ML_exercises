"""
Dataset references:
http://pjreddie.com/media/files/mnist_train.csv
http://pjreddie.com/media/files/mnist_test.csv
"""

from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import Dataset
import numpy as np

torch.manual_seed(1337)


class MnistMlp(torch.nn.Module):

    def __init__(self, inputnodes: int, hiddennodes: int, outputnodes: int) -> None:
        super().__init__()

        # number of nodes (neurons) in input, hidden, and output layer
        self.wih = torch.nn.Linear(in_features=inputnodes, out_features=hiddennodes)
        self.who = torch.nn.Linear(in_features=hiddennodes, out_features=outputnodes)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.wih(x)
        out = self.activation(out)
        out = self.who(out)
        return out


class MnistDataset(Dataset):

    def __init__(self, filepath: Path) -> None:
        super().__init__()

        self.data_list = None
        with open(filepath, "r") as f:
            self.data_list = f.readlines()

        self.features = []
        self.targets = []
        for record in self.data_list:
            all_values = record.split(",")
            features = np.asfarray(all_values[1:])
            target = int(all_values[0])
            self.features.append(features)
            self.targets.append(target)

        self.features = torch.tensor(np.array(self.features), dtype=torch.float) / 255.0
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


if __name__ == "__main__":
    # Device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # NN architecture:
    # number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # learning rate is 0.1
    learning_rate = 0.1
    # batch size
    batch_size = 10
    # number of epochs
    epochs = 3

    # Load mnist training and testing data CSV file into a datasets
    train_dataset = MnistDataset(filepath="C:\\Users\\Admin\\Downloads\\mnist_train.csv")
    test_dataset = MnistDataset(filepath="C:\\Users\\Admin\\Downloads\\mnist_test.csv")
    # Make data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # Define NN
    model = MnistMlp(inputnodes=input_nodes,
                     hiddennodes=hidden_nodes,
                     outputnodes=output_nodes)
    # Number of parameters in the model
    print(f"# Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model = model.to(device=device)

    # Define Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    ##### Training! #####
    model.train()
    for epoch in range(epochs):
        for batch_idx, (features, target) in enumerate(train_loader):
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(features), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    ##### Testing! #####
    model.eval()
    test_loss = 0
    correct = 0
    with torch.inference_mode():
        for features, target in test_loader:
            features, target = features.to(device), target.to(device)
            output = model(features)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
        classification_report

    ##### Calculate Metrics #####

    predicted_labels = []
    true_labels = []

    with torch.inference_mode():
        for features, target in test_loader:
            features, target = features.to(device), target.to(device)
            output = model(features)
            pred = output.argmax(dim=1)
            predicted_labels.extend(pred.cpu().numpy())
            true_labels.extend(target.cpu().numpy())

    # Порахуємо метрики
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision_per_class = precision_score(true_labels, predicted_labels, average=None)
    recall_per_class = recall_score(true_labels, predicted_labels, average=None)
    f1_score_per_class = f1_score(true_labels, predicted_labels, average=None)
    confusion = confusion_matrix(true_labels, predicted_labels)
    classification_rep = classification_report(true_labels, predicted_labels, target_names=[str(i) for i in range(10)])
    # Порахуємо Accuracy per class
    accuracy_per_class = []
    for class_label in range(10):
        class_indices = np.where(np.array(true_labels) == class_label)
        class_true_labels = np.array(true_labels)[class_indices]
        class_predicted_labels = np.array(predicted_labels)[class_indices]
        class_accuracy = accuracy_score(class_true_labels, class_predicted_labels)
        accuracy_per_class.append(class_accuracy)

    # Выведемо результати
    print("Accuracy:", accuracy)
    print("Overall Precision:", precision_score(true_labels, predicted_labels, average='weighted'))
    print("Overall Recall:", recall_score(true_labels, predicted_labels, average='weighted'))
    print("Overall F1-score:", f1_score(true_labels, predicted_labels, average='weighted'))
    print("Accuracy per class:", accuracy_per_class)
    print("Precision per class:", precision_per_class)
    print("Recall per class:", recall_per_class)
    print("F1-score per class:", f1_score_per_class)
    print("Confusion Matrix:")
    print(confusion)
    print("Classification Report:")
    print(classification_rep)

    ##### Save Model! #####
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(model.state_dict(), "mnist_001.pth")
