import torch
import torch.nn.functional as F
import torchvision
from loguru import logger
from torch.hub import load_state_dict_from_url
from torch.utils.data.dataloader import DataLoader

from data.imagenet import load_data
from data.transform import query_transform
from models.alexnet import AlexNet

logger.add('logs/cifar-10-consolidate.log')
device = torch.device('cuda:2')

root = '/data2/suqinghang/Dataset/Imagenet100'
batch_size = 32
train_dataloader, _, _ = load_data(root=root, batch_size=batch_size, workers=8)

test_set = torchvision.datasets.CIFAR10('/data2/suqinghang/Dataset/cifar-10', train=False, download=False, transform=query_transform())
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=8)

ori_model_path = 'checkpoints/ori-model.t'
inc_model_path = 'checkpoints/inc-model.t'
ori_model = torch.load(ori_model_path, map_location='cpu')
ori_model.to(device)
ori_model.eval()
inc_model = torch.load(inc_model_path, map_location='cpu')
inc_model.to(device)
inc_model.eval()

con_model = AlexNet(num_classes=10)# random init
state_dict = con_model.state_dict()
state_dict_pretrained = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
con_model.load_state_dict(state_dict_pretrained, strict=False)
con_model.to(device)

optimizer = torch.optim.Adam(con_model.parameters(), lr=1e-5)


best_acc = 0.0
max_epoch = 100
for epoch in range(max_epoch):

    con_model.eval()
    acc = 0.0
    correct = 0
    for img, label in test_loader:
        img, label = img.to(device), label.to(device)
        y = con_model(img)
        _, pred = torch.max(y, 1)
        correct += (pred == label).sum().data.item()
    acc = correct / len(test_loader.dataset.data)
    if acc>best_acc:
        best_acc = acc
        torch.save(con_model.cpu(), 'checkpoints/con_model.t')
        con_model.to(device)
    logger.info('{}/{} acc:{:.4f} best-acc:{:.4f}'.format(epoch, max_epoch, acc, best_acc))
    con_model.train()

    for img, label, index in train_dataloader:
        optimizer.zero_grad()
        img, label = img.to(device), label.to(device)
        
        y_old = ori_model(img) 
        normalize_y_old = y_old - torch.mean(y_old, dim=1).view(-1, 1)
        y_new = inc_model(img)
        normalize_y_new = y_new - torch.mean(y_new, dim=1).view(-1, 1)
        y_o = torch.cat((normalize_y_old, normalize_y_new), dim=1)

        y = con_model(img)

        loss = F.mse_loss(y, y_o)
        loss.backward()
        optimizer.step()
    



