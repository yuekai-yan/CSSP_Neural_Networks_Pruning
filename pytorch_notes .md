# 章节
1. [创建Dataset](#创建Dataset)
1. [搭建Neural_Network](#搭建Neural_Network)



## 创建 Dataset
```python
import os
from torch.utils.data import Dataset
from PIL import Image


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)
```
调用

```python
root_dir = "dataset/train"
ants_label_dir = "ants"

ants_dataset = MyData(root_dir, ants_label_dir)    # 按位置传参， "dataset/train" -> root_dir , "ants" -> label_dir
```

## 搭建Neural_Network

```python
import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()

        self.conv1 = Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = MaxPool2d(2)

        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(2)

        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)

        self.flatten = Flatten()

        self.linear1 = Linear(1024, 64)
        self.linear2 = Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.maxpool3(x)

        x = self.flatten(x)

        x = self.linear1(x)
        x = self.linear2(x)

        return x
```

等价于利用`nn.Sequential

```python
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x
```