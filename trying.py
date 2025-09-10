import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1307], std=[0.3081])  # 单通道
])

class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.transform = transform

        # csv文件的内容分成两列，第一列是图像路径，第二列是对应的标签
        #### 补充代码：使用 pandas 读取 CSV 文件，并将路径、标签提取到列表中
        df = pd.read_csv(csv_file,sep=';')
        self.image_paths = df['image_path'].tolist()
        self.labels = df['label'].tolist()

        #### 补充代码：利用sklearn的LabelEncoder对self.labels（string标签）编码为int
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)
        
    def __len__(self):
        #### 补充代码：返回样本个数
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        #### 补充代码：返回第idx张图像（注意不是路径）及其对应的标签，图像需要经过self.transform处理
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)   #更改数据格式为pytorch格式

        return image, label

batch_size = 64

# transform是一个方法，用来对数据进行预处理，可通过定义transform方法实现更加复杂的数据处理，如转换为tensor、图像resize、归一化等等
train_dataset = datasets.MNIST(
    root='./data', 
    train=True, 
    transform=transform,  # 使用预定义的transform
    download=True
)
test_dataset = datasets.MNIST(
    root='./data', 
    train=False, 
    transform=transform,
    download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


print("训练集样本数: ", len(train_dataset))
print("测试集样本数: ", len(test_dataset))
print("图像形状: ", train_dataset[0][0].shape)
print("标签示例: ", train_dataset[0][1])


def denormalize(img):
    mean = np.array([0.1307])
    std = np.array([0.3081])
    img = img * std[:, None, None] + mean[:, None, None]
    return np.clip(img, 0, 1)

random_indices = random.sample(range(len(train_dataset)), 4)    #随机抽取4个图像

fig, axes = plt.subplots(1, 4, figsize=(12, 4)) #画图

for i, idx in enumerate(random_indices):    #显示图像
    image, label = train_dataset[idx]
    image = denormalize(image.numpy())
    image = np.transpose(image, (1, 2, 0))
    
    axes[i].imshow(image)
    axes[i].set_title(f"Label: {label}")
    axes[i].axis("off")

plt.show()


class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        # 定义第一个卷积层：输入通道1，输出通道32，卷积核3x3，步长1，填充1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # 定义第二个卷积层：输入通道32，输出通道64，卷积核3x3，步长1，填充1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 定义第三个卷积层：输入通道64，输出通道128，卷积核3x3，步长1，填充1
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # 定义一个全连接层：输入特征128*8*8（假设输入大小为28x28），输出特征256
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        # 定义输出层：输入特征256，输出特征num_classes
        self.fc2 = nn.Linear(512, num_classes)



    def forward(self, x):
        # 第一个卷积层，随后使用ReLU激活函数和最大池化
        x = torch.nn.functional.relu(self.conv1(x)) #卷积后激活函数
        x = torch.nn.functional.max_pool2d(x, 2)    #池化层缩小数据
        
        # 第二个卷积层，随后使用ReLU激活函数和最大池化
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        
        # 第三个卷积层，随后使用ReLU激活函数和最大池化
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        
        # 展平操作
        x = x.view(x.size(0), -1)
        
        # 第一个全连接层，随后使用ReLU激活函数
        x = torch.nn.functional.relu(self.fc1(x))
        
        # 输出层
        x = self.fc2(x)
        
        # 返回输出
        return x


num_classes = 10

# 初始化模型
model = CNNModel(num_classes=num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()   #loss
optimizer = optim.Adam(model.parameters(), lr=0.001)    #优化器，自动调整学习率

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# %% [markdown]
# ### Step 3: 训练模型

# %%
num_epochs = 8
best_test_acc = 0.0  # 初始化最佳测试准确率
best_model_wts = None  # 用于保存最佳模型的权重

for epoch in range(num_epochs):
    model.train()   #模型训练
    running_loss = 0.0  #总损失
    correct = 0 #正确个数
    total = 0   #总运算个数

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")    #？

    for images, labels in train_loader_tqdm:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images) #前向传播
        loss = criterion(outputs, labels)   #计算损失
        loss.backward() #反向传播
        optimizer.step()    #更新维度

        running_loss += loss.item()

        # 计算训练正确率
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        train_loader_tqdm.set_postfix(loss=running_loss / (train_loader_tqdm.n + 1), acc=correct / total)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # 测试阶段
    model.eval()    #模型效果检测
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        test_loader_tqdm = tqdm(test_loader, desc=f"Testing")
        for images, labels in test_loader_tqdm:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            test_loader_tqdm.set_postfix(loss=test_loss / (test_loader_tqdm.n + 1), acc=correct / total)

    test_loss /= len(test_loader)
    test_acc = correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # 保存最佳模型
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_model_wts = model.state_dict()

# 保存最佳模型的权重
torch.save(best_model_wts, 'best_model.pth')

print("Training complete!")

# %%
# 加载并测试最佳模型
best_model = CNNModel(num_classes)
best_model.load_state_dict(torch.load('best_model.pth'))
best_model = best_model.to(device)

best_model.eval()
correct = 0
total = 0
test_loss = 0.0


with torch.no_grad():
    test_loader_tqdm = tqdm(test_loader, desc="Testing Best Model")
    for images, labels in test_loader_tqdm:
        images, labels = images.to(device), labels.to(device)

        outputs = best_model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    test_loss /= len(test_loader)
    test_acc = correct / total
    print(f"Best Model Test Loss: {test_loss:.4f}, Best Model Test Acc: {test_acc:.4f}")


image_paths = r"C:\Users\35023\Desktop\word\大二下课程\创新实践\data\number\five_2.jpg"
img = Image.open(image_paths).convert('L')  # 转为灰度图
img.show() 

img_tensor = transform(img).unsqueeze(0).to(device)  # 增加batch维度并移至设备

best_model.eval()  # 使用最佳模型
with torch.no_grad():
    output = best_model(img_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]  # 转换为概率
    predicted_class = torch.argmax(output).item()  # 获取预测类别
    
    # 打印清晰的结果
    print("\n图像预测结果:")
    print(f"预测数字: {predicted_class}")
    print("各类别概率分布:")
    for i, prob in enumerate(probabilities.cpu().numpy()):
        print(f"数字 {i}: {prob*100:.2f}%")