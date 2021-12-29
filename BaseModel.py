import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.hub import load_state_dict_from_url
from torch.utils.data.dataloader import DataLoader

from data.transform import query_transform, train_transform
from models.alexnet import AlexNet
from loguru import logger

is_ori = False
logger.add('logs/cifar-10-{}-5.log'.format('ori' if is_ori else 'inc'))
train_set = torchvision.datasets.CIFAR10('/data2/suqinghang/Dataset/cifar-10', train=True, download=False, transform=train_transform())
train_data =  np.array(train_set.data)
train_targets = np.array(train_set.targets)
test_set = torchvision.datasets.CIFAR10('/data2/suqinghang/Dataset/cifar-10', train=False, download=False, transform=query_transform())
test_data =  np.array(test_set.data)
test_targets = np.array(test_set.targets)
if is_ori: 
    train_set.data    = train_data[train_targets<5]
    train_set.targets = train_targets[train_targets<5]
    test_set.data     = test_data[test_targets<5]
    test_set.targets  = test_targets[test_targets<5]
else: 
    train_set.data    = train_data[train_targets>=5]
    train_set.targets = train_targets[train_targets>=5]-5
    test_set.data     = test_data[test_targets>=5]
    test_set.targets  = test_targets[test_targets>=5]-5

trainloader = DataLoader(
    dataset=train_set,
    batch_size=32,
    shuffle=True,
    num_workers=8, 
)
testloader = DataLoader(
    dataset=test_set,
    batch_size=32,
    shuffle=False,
    num_workers=8,
)

model = AlexNet(num_classes=5)
state_dict = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
model.load_state_dict(state_dict, strict=False)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda:2')
model.to(device)

best_acc = 0.0

max_epoch = 100
for epoch in range(max_epoch):
    for img, label in trainloader:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()
        y = model(img)
        loss = criterion(y, label)
        loss.backward()
        optimizer.step()
    
    model.eval()
    acc = 0.0
    correct = 0
    for img, label in testloader:
        img, label = img.to(device), label.to(device)
        y = model(img)
        _, pred = torch.max(y, 1)
        correct += (pred == label).sum().data.item()
    acc = correct / len(testloader.dataset.data)
    if acc>best_acc:
        best_acc = acc
        torch.save(model.cpu(), 'checkpoints/{}-model.t'.format('ori' if is_ori else 'inc'))
        model.to(device)
    logger.info('{}/{} acc:{:.4f} best-acc:{:.4f}'.format(epoch, max_epoch, acc, best_acc))
    model.train()


