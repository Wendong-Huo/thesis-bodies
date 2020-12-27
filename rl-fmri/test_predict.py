import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np

import common.utils as utils

class MyDataset(Dataset):
    def __init__(self, shuffle_seed=0, is_train=True):
        idx_0 = 0
        idx_100 = 100
        if not is_train:
            idx_0 += 1
            idx_100 += 1
        print(f"loading exp_0/fMRI_data_{idx_0}.npy...")
        x_0 = np.load(f"exp_0/fMRI_data_{idx_0}.npy")
        # x_0 = np.zeros(shape=[10,256], dtype=np.float32)
        y_0 = np.zeros(shape=(len(x_0)))
        print(f"loading exp_0/fMRI_data_{idx_100}.npy...")
        x_1 = np.load(f"exp_0/fMRI_data_{idx_100}.npy")
        # x_1 = np.ones(shape=[10,256], dtype=np.float32)
        x = np.concatenate((x_0, x_1))
        y = np.zeros(shape=(len(x)), dtype=np.int64)
        y[len(x_0):] = 1
        idx = np.arange(start=0, stop=len(x))
        
        np.random.seed(shuffle_seed)
        p = np.random.permutation(len(x))
        self.x, self.y, self.idx = x[p], y[p], idx[p]
        self.y = self.y.astype(np.int64)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = (self.x[idx], self.y[idx], self.idx[idx])
        return sample

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(256, 2)
        # self.fc2 = nn.Linear(2, 2)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, idx in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            wrong_items, _ = np.where(pred.eq(target.view_as(pred)).numpy()==False)
            idx = idx.numpy()
            print("wrong predictions usually happen at the beginning of an episode.")
            print(sorted(idx[wrong_items]))

    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{ len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000000, metavar='N',
    #                     help='input batch size for testing (default: 1000000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_kwargs = {'batch_size': args.batch_size}
    # test_kwargs = {'batch_size': args.test_batch_size}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_train = MyDataset(shuffle_seed=0, is_train=True)
    ds_test = MyDataset(shuffle_seed=0, is_train=False)

    train_loader = DataLoader(ds_train, shuffle=True, **train_kwargs)
    test_loader = DataLoader(ds_test, shuffle=False, batch_size=len(ds_test))
    model = Net()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
