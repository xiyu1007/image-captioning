import torch
from torch import nn
import torchvision

from colorama import init, Fore

# 初始化 colorama 库以兼容 Windows 和其他平台
init()

from utils import PathChecker

pathchecker = PathChecker()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置下载路径
models_download_path = 'Models/ResNet'
models_download_path = pathchecker.check_path_exists(models_download_path, True)
torch.hub.set_dir(models_download_path)


class Encoder(nn.Module):
    """
    Encoder: 编码器
    """

    def __init__(self, encoded_image_size=14):
        # 初始化函数，设置图像编码模型的参数
        super(Encoder, self).__init__()
        # 设置编码后的图像大小
        self.encoded_image_size = encoded_image_size

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
        输入图像大小为 image_size x image_size。
        ResNet-101 进行特征提取后，特征图大小为 (2048, image_size/32, image_size/32)。
        自适应平均池化调整特征图大小为 (2048, encoded_image_size, encoded_image_size)。
        最终输出调整维度后为 (batch_size, encoded_image_size, encoded_image_size, 2048)。
        因此，模型的输出大小为 (batch_size, encoded_image_size, encoded_image_size, 2048)。

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """

        out = self.resnet(images)  # 使用ResNet进行特征提取，输出大小为 (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # 使用自适应平均池化，将特征图调整为 (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # 调整维度顺序，将通道维度移到最后，输出为 (batch_size, encoded_image_size, encoded_image_size, 2048)
        """
        ResNet-101是一个深度卷积神经网络，具有很强的特征提取能力。
        在这个模型中，最后的全连接层被移除，因此输出的维度是一个张量，而不是一个向量。
        这个张量的最后一个维度有2048个特征通道。
        """
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
            # for c in list(self.resnet.children())[4:]:
            for c in list(self.resnet.children())[4:]:
                for p in c.parameters():
                    p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    注意力网络。
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: 编码后的图像特征大小
        :param decoder_dim: 解码器的 RNN 大小
        :param attention_dim: 注意力网络的大小
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
        前向传播。

        :param encoder_out: 编码后的图像特征，维度为 (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: 上一个解码器输出，维度为 (batch_size, decoder_dim)
        :return: 注意力加权编码，权重
        """
        # 使用线性层将编码后的图像特征映射到注意力网络的维度
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        # 使用线性层将解码器的输出映射到注意力网络的维度
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        # 计算注意力权重
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        """
        将 att1 和 att2.unsqueeze(1) 相加的操作是因为注意力机制的设计。
        通常在注意力机制中，我们希望结合两个来源的信息来计算注意力权重。
        att1 是来自编码器输出的注意力信息，
        att2.unsqueeze(1) 是来自解码器输出的注意力信息。
        通过将它们相加，我们允许模型在计算注意力权重时同时考虑这两个信息源的贡献。
        这种相加的操作在注意力机制中是一种常见的方法，它允许模型灵活地结合不同来源的信息来计算最终的注意力权重。
        """
        # 使用 softmax 计算权重值
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        # 计算注意力加权编码
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

    """
    
    attention_weighted_encoding 是注意力加权编码，
    它是根据注意力权重 alpha 对编码器输出 encoder_out 进行加权求和得到的。
    这个编码包含了编码器输出中在当前解码器状态下应该被关注的信息，
    它在生成解码器的下一个时间步的输出时起着重要作用。
    
    alpha 是注意力权重，它表示了模型在生成当前解码器输出时对输入图像各个位置的关注程度。
    这个权重告诉我们在生成当前解码器输出时，模型对输入图像的哪些部分更关注，哪些部分相对不那么重要。
    """


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    初始化方法 (__init__)：
    接受多个参数，包括 attention 网络的大小、嵌入层的大小、解码器的 RNN 大小、词汇表的大小等。
    定义了注意力网络 (self.attention)，嵌入层 (self.embedding)，dropout 层 (self.dropout)，LSTMCell 解码器 (self.decode_step) 以及其他线性层。
    调用 init_weights 方法，用均匀分布初始化一些参数。

    初始化权重方法 (init_weights)：
    用均匀分布初始化嵌入层的权重，将线性层的偏置设置为0，用均匀分布初始化线性层的权重。

    加载预训练嵌入方法 (load_pretrained_embeddings)：
    接受预训练的嵌入，将嵌入层的权重设置为预训练的嵌入。

    微调嵌入层方法 (fine_tune_embeddings)：
    接受一个布尔值，决定是否允许对嵌入层进行微调。

    初始化隐藏状态方法 (init_hidden_state)：
    接受编码的图像，创建解码器 LSTM 的初始隐藏状态和细胞状态。

    前向传播方法 (forward)：
    接受编码的图像 (encoder_out)、编码的字幕 (encoded_captions) 和字幕长度 (caption_lengths)。
    进行注意力加权，嵌入字幕，初始化 LSTM 状态。
    在每个时间步解码，生成词预测分数和注意力权重。
    返回词汇表的分数、排序后的编码字幕、解码长度、注意力权重以及排序索引。
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        :param attention_dim: attention网络的大小
        :param embed_dim: 嵌入层的大小
        :param decoder_dim: 解码器的RNN大小
        :param vocab_size: 词汇表的大小
        :param encoder_dim: 编码图像的特征大小，默认为2048
        :param dropout: dropout的比例
        """
        super(DecoderWithAttention, self).__init__()

        # 保存参数
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        """
        attention_dim: 注意力网络的大小。这个大小决定了注意力机制的复杂度和表达能力。
        通常，注意力网络的大小应该足够大，以便能够捕获输入图像和解码器输出之间的复杂关系，但也要考虑计算效率。

        embed_dim: 嵌入层的大小。这个大小决定了词嵌入的维度。词
        嵌入是将离散的词汇映射到连续的向量空间中，因此嵌入层的大小需要足够大，以便能够捕获词汇之间的语义关系。
        
        decoder_dim: 解码器的RNN大小。这个大小决定了解码器RNN单元的隐藏状态的维度。
        解码器RNN的隐藏状态用于捕获序列中的上下文信息，并生成下一个词汇的概率分布。
        
        vocab_size: 词汇表的大小。这个大小决定了模型的输出空间，即词汇表中词汇的数量。
        输出层的大小应该等于词汇表的大小，以便能够生成正确的词汇。
        
        encoder_dim: 编码图像的特征大小。这个大小是编码器输出的特征向量的维度。
        在这里，使用了默认值2048，这是因为通常情况下，在使用预训练的卷积神经网络（如ResNet）提取图像特征时，最后一层的特征维度通常是2048。
        """

        # 定义注意力网络
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        # 定义嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 定义dropout层
        self.dropout = nn.Dropout(p=self.dropout)

        # 定义解码的LSTMCell
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)

        # 定义线性层以找到LSTMCell的初始隐藏状态
        self.init_h = nn.Linear(encoder_dim, decoder_dim)

        # 定义线性层以找到LSTMCell的初始细胞状态
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # 定义线性层以创建一个sigmoid激活的门
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        # 定义sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

        # 定义线性层以在词汇表上找到分数
        self.fc = nn.Linear(decoder_dim, vocab_size)

        # 初始化一些层的权重
        self.init_weights()

    def init_weights(self):
        """
        用均匀分布的值初始化一些参数，以便更容易地进行收敛。
        """
        # 初始化嵌入层的权重，使用均匀分布在(-0.1, 0.1)之间
        self.embedding.weight.data.uniform_(-0.1, 0.1)

        # 初始化线性层的偏置，将所有元素设置为0
        self.fc.bias.data.fill_(0)

        # 初始化线性层的权重，使用均匀分布在(-0.1, 0.1)之间
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        使用预训练的嵌入加载嵌入层。

        :param embeddings: 预训练的嵌入
        """
        # 将嵌入层的权重设置为预训练的嵌入
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        允许微调嵌入层吗？（如果使用预训练的嵌入，不允许微调是有意义的。）
        :param fine_tune: 是否允许微调
        """
        # 设置嵌入层的requires_grad属性，以决定是否允许微调
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        根据编码的图像创建解码器的LSTM的初始隐藏状态和细胞状态。

        :param encoder_out: 编码的图像，维度为 (batch_size, num_pixels, encoder_dim)
        :return: 隐藏状态，细胞状态
        """
        # 对编码的图像进行平均，得到 (batch_size, encoder_dim) 的张量
        mean_encoder_out = encoder_out.mean(dim=1)

        # 使用线性层找到LSTM的初始隐藏状态
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)

        # 使用线性层找到LSTM的初始细胞状态
        c = self.init_c(mean_encoder_out)

        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        前向传播。

        :param encoder_out: 编码的图像，维度为 (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: 编码的字幕，维度为 (batch_size, max_caption_length)
        :param caption_lengths: 字幕长度，维度为 (batch_size, 1)
        :return: 词汇表的分数，排序后的编码字幕，解码长度，权重，排序索引

        预测分数 (predictions)：
        大小为 (batch_size, max(decode_lengths), vocab_size)。
        表示模型对词汇表中每个单词的预测分数。

        排序后的编码字幕 (encoded_captions)：
        大小为 (batch_size, max_caption_length)。
        表示输入的编码字幕，按照字幕长度降序排列。

        解码长度 (decode_lengths)：
        一个包含每个样本对应的解码长度的列表，大小为 (batch_size,)。
        表示每个样本生成字幕的实际长度。

        注意力权重 (alphas)：
        大小为 (batch_size, max(decode_lengths), num_pixels)。
        表示模型在生成每个词时对输入图像中不同部分的关注程度。

        排序索引 (sort_ind)：
        大小为 (batch_size,)。
        表示对输入数据按字幕长度降序排列后的索引。
        """

        # 获取维度信息
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # 将图像展平
        # input img = torch.Size([1, 3, 256, 256])
        # encoder_out :(batch_size, encoded_image_size, encoded_image_size, 2048)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        # encoder_out :(batch_size, encoded_image_size*encoded_image_size, 2048)

        num_pixels = encoder_out.size(1)  # encoded_image_size*encoded_image_size = 14 * 14

        # 按长度降序排列输入数据

        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        # sort_id 排序前的索引
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # 嵌入层
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # 初始化LSTM状态
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # 由于在生成<end>后我们就完成了生成，所以我们不会在<end>位置解码
        # 因此，解码长度实际上是长度 - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # 创建张量以保存词预测分数和注意力权重
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # 在每个时间步进行解码
        # 通过基于解码器先前隐藏状态输出的注意力权重来加权编码器的输出
        # 然后使用先前的单词和注意力加权编码生成解码器中的新单词

        for t in range(max(decode_lengths)):
            # 计算当前时间步的批次大小
            batch_size_t = sum([l > t for l in decode_lengths])
            # decode_lengths已经降序
            """
            batch_size_t 在这段代码中的作用是确定当前时间步（t）的批次大小（batch size）。
            让我用通俗的语言解释一下这段代码的功能和 batch_size_t 的作用：
            在这段代码中，有一个循环 for t in range(max(decode_lengths))，它迭代了解码长度的最大值次数。
            这个解码长度通常是指解码器生成的句子的最大长度。在每个时间步（t）中，我们需要根据当前时间步的情况来确定批次的大小。
            在计算 batch_size_t 时，它统计了当前时间步（t）中，有多少个句子仍然在解码中，也就是说，这个时间步还没有结束。
            这是通过检查每个句子的解码长度是否大于当前时间步（t）来实现的。如果某个句子的解码长度大于当前时间步（t），
            那么它仍然处于解码过程中，就会被计算在 batch_size_t 中。
            然后，在这个时间步（t），我们根据当前的批次大小 batch_size_t，
            从编码器输出 encoder_out 和隐藏状态 h 中选择对应的批次数据，以便在注意力模型中使用。这样做的目的是，
            为了在每个时间步上仅处理仍然在解码的句子，而不是整个批次中的所有句子。
            """

            # 使用注意力模型计算注意力加权的编码器输出和注意力权重
            # h,encoder_out => batch_size, encoded_image_size * encoded_image_size, 2048
            # encoder_out[:batch_size_t] => 0:batch_size_t, encoded_image_size * encoded_image_size, 2048
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            # 计算门控标量，用于调整注意力加权的编码器输出
            """
            self.f_beta()：这是一个神经网络层，它将隐藏状态 h[:batch_size_t] 映射到与编码器输出的维度相同的空间（encoder_dim）。
            这个映射是为了与编码器输出进行加权求和。
            """
            # 定义线性层以创建一个sigmoid激活的门
            # self.f_beta = nn.Linear(decoder_dim, encoder_dim)
            # 解码器的隐藏状态被调整为与编码器输出相同的维度
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding

            """
            embeddings => [batch_size, max_caption_length, embedding_dim]
            attention_weighted_encoding => (batch_size, encoder_dim)
            """

            # 使用LSTMCell解码器进行一步解码
            # print(Fore.BLUE + str(embeddings.shape))  # torch.Size([1, 52, 512])
            # print(Fore.BLUE + str(attention_weighted_encoding.shape))  # torch.Size([1, 2048])
            # print(Fore.BLUE + str(q.shape))  # torch.Size([1, 2560])
            """
            embeddings 的维度是 [1, 52, 512]，表示一个大小为 1 的批次，每个序列有 52 个时间步，每个时间步有 512 维的嵌入向量。
            attention_weighted_encoding 的维度是 [1, 2048]，表示一个大小为 1 的批次，每个样本有 2048 维的编码向量。
            根据 torch.cat 的作用，它将 embeddings[:batch_size_t, t, :] 和 attention_weighted_encoding 
            在第 1 维度上拼接，因此结果张量 q 的维度是 [1, 2560]，其中 2560 是 512（来自嵌入向量）加上 2048（来自注意力加权编码向量）的总维度。
            这就解释了为什么 q.shape 的输出是 torch.Size([1, 2560])。
            """

            """
            embeddings: 是一个三维张量，形状为 [1, 52, 512]，其
            中 1 表示批次大小，52 表示每个序列有 52 个时间步，512 表示每个时间步的嵌入维度。
            [:batch_size_t, t, :]: 这是切片操作。[:batch_size_t] 
            表示取 0 到 batch_size_t-1 的批次索引，[t] 表示取第 t 个时间步，[:] 表示取所有的嵌入维度。
            综合起来，embeddings[:batch_size_t, t, :] 
            表示从 embeddings 张量中选择批次索引为 0 到 batch_size_t-1、时间步索引为 t、嵌入维度为所有的部分。
            这实际上是一个张量的切片操作，返回的结果是一个形状为 [batch_size_t, 512] 的二维张量，
            其中包含了选定批次、时间步和嵌入维度的部分
            
            """

            """
            在这种情况下，decoder.decode_step 是一个 LSTM 单元，它的输入是一个张量和一个包含隐藏状态 (h, c) 的元组。
            在 PyTorch 中，LSTM 单元的 forward 方法期望的输入是
            当前时间步的输入以及前一个时间步的隐藏状态 (h_t-1, c_t-1)。
            在你的代码中，你可能已经初始化了 h 和 c 作为 LSTM 单元的初始隐藏状态，
            然后在每个时间步都更新它们。因此，你可以直接将 (h, c) 作为输入提供给 decoder.decode_step。
            当你调用 decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c)) 时，
            PyTorch 会将当前时间步的输入和前一个时间步的隐藏状态传递给 LSTM 单元，
            然后返回下一个时间步的输出和更新后的隐藏状态 (h_t, c_t)。
            """
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)

            # 通过全连接层生成词预测分数
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            # 将预测分数存储到predictions张量中

            predictions[:batch_size_t, t, :] = preds
            # 将注意力权重存储到alphas张量中
            alphas[:batch_size_t, t, :] = alpha
        # 返回模型的输出：词预测分数、排序后的编码字幕、解码长度、注意力权重、排序索引
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
