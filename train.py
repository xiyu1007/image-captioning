import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu

from colorama import init, Fore

# 初始化 colorama 库以兼容 Windows 和其他平台
init()

# Data parameters
# output_path_json_hdf5
data_folder = 'out_data/img_json_hdf5'  # folder with data files saved by create_input_files.py
data_name = 'flickr30k_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 512  # dimension of word embeddings（词嵌入的维度）
attention_dim = 512  # dimension of attention linear layers（注意力机制中线性层的维度）
decoder_dim = 512  # dimension of decoder RNN（解码器RNN的维度）
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
# （设置模型和PyTorch张量的设备，如果有CUDA则使用GPU，否则使用CPU）
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lots of computational overhead
# （仅当模型的输入具有固定大小时设置为true；否则会有很多计算开销）


# Training parameters
# 训练参数
start_epoch = 0  # 开始的训练轮次
epochs = 2  # 训练的总轮次
epochs_since_improvement = 0  # 自上次在验证集上取得改进以来的轮次数，用于提前停止
batch_size = 30  # 32 每个训练批次中的样本数
workers = 0  # 数据加载的工作进程数 num_workers参数设置为0，这将使得数据加载在主进程中进行，而不使用多进程。
# 这个错误是由于h5py对象无法被序列化（pickled）引起的。
# 在使用多进程（multiprocessing）加载数据时，数据加载器（DataLoader）会尝试对每个批次的数据进行序列化，以便在不同的进程中传递。
encoder_lr = 1e-4  # 编码器的学习率（如果进行微调）
decoder_lr = 4e-4  # 解码器的学习率
grad_clip = 5.  # 梯度裁剪的阈值，用于防止梯度爆炸
alpha_c = 1.  # '双重随机注意力'的正则化参数
best_bleu4 = 0.  # 当前的最佳 BLEU-4 分数
print_freq = 100  # 每训练多少个批次打印一次训练/验证统计信息
fine_tune_encoder = False  # 是否对编码器进行微调
checkpoint = None  # 检查点的路径，如果为 None，则没有检查点


def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None:
        # LSTM
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        # ResNet
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    """
    normalized_pixel_value = (original_pixel_value - mean[C]) / std[C]
    在这里给出的参数是根据 ImageNet 数据集的均值和标准差计算得到的。
    
    transforms.Normalize 
    通常用于标准化图像的像素值。在这里，给定了均值 (mean) 和标准差 (std)，
    它将图像的每个通道进行标准化，使其均值为0，标准差为1。
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # 创建一个图像变换组合，包含 normalize 变换
    # transform = transforms.Compose([normalize])
    # TODO num_workers = 0
    """torch.utils.data.DataLoader：PyTorch 的数据加载器类，用于按批次加载数据。 
    CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize]))： 
    这是一个自定义的数据集类 CaptionDataset 的实例化。
    该数据集用于加载训练数据， transform 参数指定了在加载数据时要应用的图像变换，这里使用了之前定义的标准化变换。 
    batch_size：指定每个批次的样本数量。 
    shuffle=True：表示在每个 epoch 开始时是否对数据进行随机洗牌。 
    这有助于确保每个批次都包含来自不同样本的数据，有助于模型的训练。 
    num_workers：表示用于加载数据的子进程的数量。这有助于加速数据加载过程。若报错改为0
    pin_memory=True：如果设为 True，数据加载器将数据加载到 
    CUDA 的固定内存区域，以便更快地将数据传递给 GPU。这在使用 GPU 训练时可以提高性能。 """
    # train_loader = torch.utils.data.DataLoader(
    #     CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize])),
    #     batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN'),
        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    # TODO
    for epoch in range(start_epoch, epochs):
        # 如果连续 20 个 epoch 都没有性能提升，则提前终止训练
        if epochs_since_improvement == 20:
            break
        # 如果经过一定 epoch 数（8 的倍数）仍然没有性能提升，则进行学习率衰减
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            # 调整解码器（decoder）的学习率，将当前学习率乘以 0.8
            adjust_learning_rate(decoder_optimizer, 0.8)

            # 如果需要对编码器（encoder）进行微调，也调整编码器的学习率
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    # 设置解码、编码器为训练模式（启用 dropout 和批归一化）
    decoder.train()
    encoder.train()

    # 用于记录前向传播和反向传播的时间的指标
    batch_time = AverageMeter()  # forward prop. + back prop. time
    # 用于记录数据加载时间的指标
    data_time = AverageMeter()  # data loading time
    # 用于记录每个单词的损失的指标
    losses = AverageMeter()  # loss (per word decoded)
    # 用于记录 top-5 准确率的指标
    top5accs = AverageMeter()  # top5 accuracy

    # start = time.time()

    # Batches
    with tqdm(total=len(train_loader), desc=f"Training:  Epoch {epoch + 1}/{epochs}") as t:
        for i, (imgs, caps, caplens) in enumerate(train_loader):
            # data_time.update(time.time() - start)
            # Move to GPU, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            # 当你创建一个模型实例，并将输入数据传递给模型时，实际上是调用了这个模型的 forward 方法。
            # 输出为(batch_size, encoded_image_size 14, encoded_image_size 14, 通道维度 2048)
            imgs = encoder(imgs)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            """
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
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]  # torch.Size([32, 51])

            # TODO
            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            # torch.Size([32, 20, 506])
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # scores = scores.reshape(-1, 506)
            # targets = targets.reshape(-1)

            # END TODO
            # Calculate loss
            loss = criterion(scores, targets)
            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Back prop.
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            if grad_clip is not None:
                clip_gradient(decoder_optimizer, grad_clip)
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, grad_clip)

            # Update weights
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            # Keep track of metrics
            top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))

            # batch_time.update(time.time() - start)
            # start = time.time()

            t.set_postfix(loss=f"{losses.val:.4f}({losses.avg:.4f})",
                          top5=f"{top5accs.val:.3f} ({top5accs.avg:.3f})")
            t.update(1)
            # Print status
            # if i % print_freq == 0:
            #     print('Epoch: [{0}][{1}/{2}]\t'
            #           'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
            #                                                                   batch_time=batch_time,
            #                                                                   data_time=data_time, loss=losses,
            #                                                                   top5=top5accs))


def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()

            # TODO
            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            # torch.Size([32, 20, 506])
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # END TODO

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            # allcaps = allcaps[sort_ind]  # because images were sorted in the decoder

            # TODO
            allcaps = allcaps.to(sort_ind.device)[sort_ind]

            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)

        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
