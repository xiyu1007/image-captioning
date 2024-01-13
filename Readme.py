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

