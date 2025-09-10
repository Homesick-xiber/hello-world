import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from PIL import Image
import torchvision.transforms as transforms
import os

# 1. 模型架构定义
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        # 简化版图像编码器，这里用一个简单的CNN代替
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 64 * 64, embed_dim) # 假设输入图像是256x256

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', embed_dim=512):
        super().__init__()
        # 使用预训练的BERT模型作为文本编码器
        self.bert = BertModel.from_pretrained(model_name)
        self.text_projection = nn.Linear(self.bert.config.hidden_size, embed_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 获取 [CLS] 标记的嵌入，代表整个句子的向量
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]
        return self.text_projection(cls_embedding)

class CLIPModel(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07) # 温度参数，用于控制对比度

    def forward(self, images, captions_ids, attention_mask):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(captions_ids, attention_mask)
        
        # 特征归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        return image_features, text_features

# 2. 对比损失函数
def contrastive_loss(image_features, text_features, temperature):
    # 计算余弦相似度
    logits = (text_features @ image_features.T) / temperature
    
    # 对称损失
    images_similarity = F.softmax(logits, dim=1)
    texts_similarity = F.softmax(logits.T, dim=1)
    
    # 构造标签
    labels = torch.arange(len(images_similarity), device=images_similarity.device)
    
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)
    
    loss = (loss_i + loss_t) / 2
    return loss

# 3. 数据集和数据加载器（这里用一个简化版）
class SimpleDataset(Dataset):
    def __init__(self, image_dir, captions):
        self.image_dir = image_dir
        self.captions = captions
        self.image_files = os.listdir(image_dir)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        caption = self.captions[idx]
        tokens = self.tokenizer(caption, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
        
        return image, tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0)

# 4. 训练循环
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 示例数据（请替换为自己的数据集）
    # 创建一个名为 'data' 的文件夹，并在其中放入一些图像文件（例如：img1.jpg, img2.jpg...）
    # 确保图像文件名和下面captions列表的顺序是对应的。
    image_dir = 'data'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        # TODO: 请在这里放入你的图像文件
    
    captions = [
        "A photo of a dog playing in the park.",
        "A car driving on a street.",
        "A beautiful sunset over the mountains."
    ]
    
    dataset = SimpleDataset(image_dir, captions)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = CLIPModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, input_ids, attention_mask in dataloader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            image_features, text_features = model(images, input_ids, attention_mask)
            
            loss = contrastive_loss(image_features, text_features, model.temperature)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train()