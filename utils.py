import os
import time
import numpy as np
import h5py
import json
import torch
import cv2
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
from colorama import init, Fore

# 初始化 colorama
init(autoreset=True)


def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    pathchecker = PathChecker()
    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()  # $

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue

        path = os.path.join(image_folder, img['filepath'], img['filename']) \
            if dataset == 'coco' else os.path.join(image_folder, img['filename'])
        path = os.path.normpath(path)

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)
    # Create word map # $ 词映射

    # 创建一个单词列表，其中包含词频大于 min_word_freq 的单词
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    # 创建一个字典，将单词映射到它们的索引（索引从1开始）
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # 为所有输出文件创建基本/根名称
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'
    output_json_path = \
        pathchecker.check_and_create_filename(os.path.join(output_folder, 'WORDMAP_' + base_filename), 'json')

    # 将词映射保存为 JSON
    with open(output_json_path, 'w') as j:
        json.dump(word_map, j)
        print(Fore.BLUE + f"成功：已创建 JSON 文件 ==> {output_json_path}")

    # 为每张图像获取样本描述，并将图像保存到 HDF5 文件，描述及其长度保存到 JSON 文件
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:
        """
        impaths:[
        'D:\\_01_python\\Image-Captioning\\dataset\\
        short_flickr30k_images_datasets\\short_flickr30k_images_datasets\\1000092795.jpg', 
        'D:\\_01_python\\Image-Captioning\\dataset\\
        short_flickr30k_images_datasets\\short_flickr30k_images_datasets\\10002456.jpg']
        imcaps:[
        [['Two', 'young', 'guys', 'with', 'shaggy', 'hair', 'look', 'at', 'their', 'hands', 'while', 'hanging', 'out', 'in', 'the', 'yard'], 
        ['Two', 'young', ',', 'White', 'males', 'are', 'outside', 'near', 'many', 'bushes'], 
        ['Two', 'men', 'in', 'green', 'shirts', 'are', 'standing', 'in', 'a', 'yard'], 
        ['A', 'man', 'in', 'a', 'blue', 'shirt', 'standing', 'in', 'a', 'garden'],
        ['Two', 'friends', 'enjoy', 'time', 'spent', 'together']]]
        """

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'w') as h:
            # 记录我们每张图像采样的描述数量
            h.attrs['captions_per_image'] = captions_per_image

            # 在 HDF5 文件中创建数据集以存储图像
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\n正在读取 %s 图像和描述，存储到文件中...\n" % split)
            time.sleep(0.01)
            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths, desc="图像处理")):
                # 采样描述
                if len(imcaps[i]) < captions_per_image:
                    # 如果此图像的现有描述数量少于 captions_per_image，
                    # 通过从现有描述中随机选择来采样额外的描述。
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    # 如果此图像的现有描述数量等于或大于 captions_per_image，
                    # 通过从现有列表中随机选择描述来采样描述。
                    captions = sample(imcaps[i], k=captions_per_image)

                # 断言检查
                assert len(captions) == captions_per_image

                # 使用 cv2 读取图像
                img = cv2.imread(impaths[i])

                # 如果需要，将 BGR 转换为 RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # 检查图像是否为灰度图（2D），如果是，将其转换为 RGB（3D）
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                # 将图像调整为 (256, 256)
                img = cv2.resize(img, (256, 256))

                # 转置维度以将通道维度放在第一个维度
                #  (height, width, channels) ==> (channels, height, width)
                img = np.transpose(img, (2, 0, 1))
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255
                # 将图像保存到 HDF5 文件中
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    # 编码标题
                    # 对标题中的每个单词进行编码，如果单词在 word_map 中不存在，则使用 <unk> 的索引
                    # 添加结束标记和填充 <pad>，确保总长度为 max_len
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            output_json_path = \
                pathchecker.check_and_create_filename(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename),
                                                      'json')
            # Save encoded captions and their lengths to JSON files
            with open(output_json_path, 'w') as j:
                json.dump(enc_captions, j)
            output_json_path = \
                pathchecker.check_and_create_filename(os.path.join(output_folder, split + '_CAPLENS_' + base_filename),
                                                      'json')
            with open(output_json_path, 'w') as j:
                json.dump(caplens, j)
# $
def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.
    该函数的目的是将嵌入张量（embedding tensor）用均匀分布的值填充。
    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """

    pathchecker = PathChecker()
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    file_path = pathchecker.process_path('./out_data/save_model/'+filename)
    torch.save(state, file_path)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        file_path = pathchecker.process_path('./out_data/save_model/'+'BEST_' + filename)
        torch.save(state, file_path)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

class PathChecker:
    def __init__(self, current_directory=None):
        """
        param current_directory: 以Image-Captioning为根目录
        """
        self.current_directory = current_directory or os.getcwd()

    def check_path_exists(self, path, create_if_not_exist=False):
        full_path = self.process_path(path)
        # 检查目录是否存在
        if os.path.exists(full_path):
            # print(Fore.BLUE + f"The directory or file '{full_path}' exists.")
            return full_path
        else:
            print(Fore.YELLOW + f"The directory or file '{full_path}' does not exist.")
            # 如果设置了create_if_not_exist为True，自动创建目录
            if create_if_not_exist:
                os.makedirs(full_path)
                print(Fore.BLUE + f"Success: Directory '{full_path}' created.")
                return full_path  # 返回True表示目录被创建
            else:
                return None  # 返回False表示目录不存在

    def check_and_create_filename(self, file_path, file_format='txt',create_new_if_exist=False):
        full_path = self.process_path(file_path)

        parent_path = os.path.dirname(full_path)
        temp_path = full_path
        extension = f'.{file_format.lower()}'

        count = 1
        if self.check_path_exists(parent_path, True):
            if not os.path.exists(f"{full_path}{extension}") or not create_new_if_exist:
                full_path = full_path + extension
            else:
                while os.path.exists(temp_path + "_" + str(count) + extension):
                    count += 1
                full_path = temp_path + "_" + str(count) + extension
            try:
                with open(full_path, 'w'):
                    # Create the file
                    return os.path.normpath(full_path)
            except Exception as e:
                print(Fore.RED + f"Error while creating {file_format} file: {e}")
        else:
            return None

    def process_path(self, path):
        # 如果路径为绝对路径，则直接返回
        if os.path.isabs(path):
            return os.path.normpath(path)
        processed_path = path
        # 如果路径以 ".." 开头，则选择上一级目录
        if path.startswith('..'):
            if path.startswith('..'):
                parent_directory = os.path.dirname(self.current_directory)
                processed_path = os.path.join(parent_directory, path.lstrip('..\\/'))
        # 如果路径以 "." 开头，则去除路径开头的点再拼接
        elif path.startswith('.'):
            processed_path = os.path.join(self.current_directory, path.lstrip('.\\/'))
        else:
            processed_path = os.path.join(self.current_directory, path)
        return os.path.normpath(processed_path)
