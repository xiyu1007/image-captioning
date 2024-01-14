"""
数据后带有 # $ 表示Readme中有其数据结构展示
"""

# $data
"""
image_name|comment_number|comment
1000092795.jpg| 0| Two young guys with shaggy hair look at their hands while hanging out in the yard .
1000092795.jpg| 1| Two young , White males are outside near many bushes .
1000092795.jpg| 2| Two men in green shirts are standing in a yard .
1000092795.jpg| 3| A man in a blue shirt standing in a garden .
1000092795.jpg| 4| Two friends enjoy time spent together .
10002456.jpg| 0| Several men in hard hats are operating a giant pulley system .
10002456.jpg| 1| Workers look down from up above on a piece of equipment .
10002456.jpg| 2| Two men working on a machine wearing hard hats .
10002456.jpg| 3| Four men on top of a tall structure .
10002456.jpg| 4| Three men on a large rig .
"""
# $json
"""
{
  "images": [
    {
      "split": "train",
      "filepath": "path/to/image1",
      "filename": "image1.jpg",
      "sentences": [
        {
          "tokens": ["a", "cat", "on", "the", "mat"],
          "raw": "A cat on the mat."
        },
        {
          "tokens": ["a", "dog", "in", "the", "yard"],
          "raw": "A dog in the yard."
        }
      ]
    },
    {
      "split": "val",
      "filepath": "path/to/image2",
      "filename": "image2.jpg",
      "sentences": [
        {
          "tokens": ["a", "bird", "on", "a", "branch"],
          "raw": "A bird on a branch."
        }
      ]
    }
}
"""
# $word_freq( Counter() )
"""
Counter({'a': 112, 'in': 52, 'A': 42, 'on': 36, 'the': 28, 'of': 27})
"""
# $word_map
"""
{'Two': 1, 'young': 2, 'guys': 3, 'with': 4,
 'at': 5, '<unk>': 67, '<start>': 68, '<end>': 69, '<pad>': 0}
"""
# $HDF5
"""
这行代码使用了HDF5（H5py库）创建了一个名为 'images' 的数据集。让我为你解释一下参数：
h：这是 HDF5 文件的对象，表示你正在向其添加数据集。
'images'：这是数据集的名称，将在 HDF5 文件中标识这个数据集。
(len(impaths), 3, 256, 256)：这是数据集的形状，表示数据集将是一个四维数组。
具体来说，它有 len(impaths) 行，每一行都是一个大小为 (3, 256, 256) 的三维数组。
这适用于图像数据，其中 3 表示通道数，256 表示高度，256 表示宽度。
dtype='uint8'：这是数据集的数据类型，表示数据集中的元素将是 8 位无符号整数。
通常在图像数据中，像素值的范围是 0 到 255，因此使用 uint8 来表示。
"""
# $init_embedding
"""
这段代码是一个用于初始化神经网络中嵌入层（Embedding Layer）的函数。让我详细解释一下：
目的： 初始化神经网络中的嵌入层的权重。嵌入层通常用于将离散的索引（如词语或类别）映射到高维的实值向量空间。
初始化方法： 使用均匀分布（Uniform Distribution）进行初始化。
初始化的数值范围是从一个小的负偏差（-bias）到一个小的正偏差（bias）。

这个偏差的计算方式是 $\sqrt{\frac{3.0}{\text{{嵌入层的维度}}}}$。
这样的初始化方法有助于确保初始权重的合理范围，促使模型更容易学到有意义的表示。
具体解释：
bias 的计算：通过公式 $\sqrt{\frac{3.0}{\text{{嵌入层的维度}}}}$ 计算初始化的偏差值。
torch.nn.init.uniform_：PyTorch提供的初始化函数，它将给定张量（在这里是嵌入层的权重）用均匀分布的随机值填充。
这里设置了填充值的上下限，即 $-bias$ 到 $bias$ 之间。
调用方法：
调用时，你需要传递一个嵌入层的权重张量（通常是一个二维的张量，其中每行对应于一个嵌入向量）作为参数。
函数会修改这个张量的值，将其填充为均匀分布的随机数。
这样的初始化方法有助于避免在训练初期出现梯度爆炸或梯度消失等问题，提高了训练的稳定性。
均匀分布是一种概率分布，其中每个数值在给定的范围内都有相等的概率被选择。
在上下限之间，所有数值的概率密度函数是常数。均匀分布的随机值填充指的是从均匀分布中随机抽取数值，
并将这些数值填充到某个数据结构（如张量或数组）中。
"""

