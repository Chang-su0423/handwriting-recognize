import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import struct
import numpy as np
from net import Net


class MNISTDataset(Dataset):
    def __init__(self, image_file, label_file):
        self.images = self.read_images(image_file)
        self.labels = self.read_labels(label_file)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    def read_images(self, file):
        with open(file, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols, 1)
        return images

    def read_labels(self, file):
        with open(file, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = image.copy()  # 确保图像是可写的
        if self.transform:
            image = self.transform(image)
        return image, label


def train():
    train_dataset = MNISTDataset('data_set/train-images-idx3-ubyte', 'data_set/train-labels-idx1-ubyte')
    test_dataset = MNISTDataset('data_set/t10k-images-idx3-ubyte', 'data_set/t10k-labels-idx1-ubyte')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

    # 将模型保存到指定路径
    torch.save(model.state_dict(), 'model/mnist_net.pth')


if __name__ == '__main__':
    train()