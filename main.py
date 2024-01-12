import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import tensorflow
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 图像处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# 加载预训练的ResNet模型
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        return features


# 定义带注意力机制的解码器RNN
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        outputs = self.softmax(outputs)
        return outputs


# 加载图像和处理
image_path = './dataset/img/img.png'
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0)

# 加载训练好的模型
encoder = EncoderCNN(embed_size=256)
decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=10000, num_layers=1)

encoder.load_state_dict(torch.load('encoder_weights.pth'))
decoder.load_state_dict(torch.load('decoder_weights.pth'))

# 设置模型为评估模式
encoder.eval()
decoder.eval()

# 图像特征提取
image_tensor = Variable(image)
feature = encoder(image_tensor)

# 生成描述
start_token = Variable(torch.LongTensor([[1]]))  # 1表示句子的开始
max_length = 20
result_caption = []

for i in range(max_length):
    caption = decoder(feature, start_token, [1])
    _, predicted = caption.max(2)
    result_caption.append(predicted.item())
    start_token = predicted

# 输出生成的描述
print(result_caption)
