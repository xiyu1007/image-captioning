import torch
from torch import nn
import torchvision
from utils import PathChecker

pathchecker = PathChecker()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置下载路径
models_download_path = 'Models/ResNet'
models_download_path = pathchecker.check_path_exists(models_download_path,True)
torch.hub.set_dir(models_download_path)

class Encoder(nn.Module):
    """
    Encoder: 编码器
    """

    def __init__(self, encoded_image_size=14):
        # 初始化函数，设置图像编码模型的参数
        super(Encoder, self).__init__()
        # 设置编码后的图像大小
        self.enc_image_size = encoded_image_size

        # # 使用预训练的 ImageNet ResNet-50 模型
        # resnet = torchvision.models.resnet50(pretrained=True)
        # 使用预训练的 ImageNet ResNet-101 模型
        # resnet = torchvision.models.resnet101(pretrained=True)
        resnet = torchvision.models.resnet101(weights=torchvision.models.resnet.ResNet101_Weights.IMAGENET1K_V1)
        """
        IMAGENET1K_V1：ResNet-101在ImageNet上的第一个版本的预训练权重。
        IMAGENET1K_V1_1：ResNet-101在ImageNet上的第一个版本的预训练权重，可能是更新或修正版本。
        IMAGENET21K：ResNet-101在更大的ImageNet-21K数据集上进行的预训练。
        DEFAULT：获取最新的预训练权重。
        """

        # 移除线性层和池化层（因为我们不进行分类任务）
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # 调整图像大小以允许输入大小可变的图像
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        """
        nn.AdaptiveAvgPool2d: 这是 PyTorch 中的自适应平均池化层，它可以将输入的任意大小的二维数据进行自适应平均池化。
        它的输出大小是固定的，由参数指定。
        (encoded_image_size, encoded_image_size): 
        这是 nn.AdaptiveAvgPool2d 层的参数，表示输出的大小。表示最终编码后的图像的大小。
        这一行代码的作用是对输入的特征图进行自适应平均池化，将其调整为指定的输出大小 (encoded_image_size, encoded_image_size)。
        这通常用于确保模型的输出是固定大小的，不受输入图像大小的影响。
        在图像编码任务中，这有助于将不同尺寸的输入图像映射到相同大小的特征图，以便后续的处理。
        """
        # 执行微调
        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # 使用ResNet进行特征提取，输出大小为 (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # 使用自适应平均池化，将特征图调整为 (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # 调整维度顺序，将通道维度移到最后，输出为 (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        允许或禁止对编码器的卷积块2到4进行梯度计算。

        :param fine_tune: 是否允许微调?
        """
        # 禁止对整个 ResNet 的梯度计算
        for p in self.resnet.parameters():
            p.requires_grad = False

        # 如果允许微调，仅允许微调卷积块2到4
        if fine_tune:
            # ResNet-50
            # for c in list(self.resnet.children())[4:]:
            for c in list(self.resnet.children())[5:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        # 通过线性层将编码后的图像特征映射到注意力网络的维度
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # 通过线性层将解码器的输出映射到注意力网络的维度
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # 通过线性层计算进行 softmax 的权重值
        self.full_att = nn.Linear(attention_dim, 1)
        # 激活函数 ReLU
        self.relu = nn.ReLU()
        # softmax 层，用于计算权重值
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
