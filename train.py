import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 准备数据集
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10("./CIFAR10", train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10("./CIFAR10", train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

print(len(train_data), len(test_data))

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 搭建网络，十分类
class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 创建网络模型
test = Test().to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rat = 0.01
optimizer = torch.optim.SGD(test.parameters(), lr=learning_rat)

# 设置训练网络的一些参数

# 训练次数
total_train_step = 0

# 测试次数
total_test_step = 0

# 训练轮数
epoch = 50

# 添加tensorboard
writer = SummaryWriter("./traintest_log")

for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i + 1))

    # 训练开始
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = test(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 ==0:
            print("训练次数:{},Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)


    # 测试
    total_test_loss = 0.0
    total_accu = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = test(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss = total_test_loss+loss.item()
            accuacy = (outputs.argmax(1)==targets).sum()
            total_accu = total_accu + accuacy

    print("整体测试集的loss:{}".format(total_test_loss))
    print("整体测试集正确率:{}".format(total_accu/len(test_data)))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuacy", total_accu/len(test_data) , total_test_step)
    total_test_step = total_test_step + 1

    torch.save(test, "cifar10_models/test_cifar10_gpu{}.pth".format(i))

writer.close()

