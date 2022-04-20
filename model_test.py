import torch
import torchvision
from PIL import Image
from torch import nn
import pickle
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


def load_file(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data

model = torch.load("cifar10_models/test_cifar10_gpu49.pth",map_location=torch.device('cpu'))
# print(model)
images_path = "imgs"
for root,ds,fs in os.walk(images_path):
    for f in fs:
        img_path  = os.path.join(root, f)
        image = Image.open(img_path)
        # print(image)
        image = image.convert('RGB')
        transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

        image = transform(image)
        # print(image.shape)

        image = torch.reshape(image, (1, 3, 32, 32))
        model.eval()
        with torch.no_grad():
            output = model(image)
        # print(output)

        result = output.argmax(1)

        data = load_file('CIFAR10/cifar-10-batches-py/batches.meta')
        # print(data.keys())
        labels = data['label_names']
        print(img_path + ":"+labels[result])
        # print(result)


