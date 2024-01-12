import json
import pandas as pd
import os
import random


class DatasetConverter:
    def __init__(self, csv_path, image_folder, output_json_path,
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        将数据集转化为json文件，数据集格式参照Readme $data。
        :param csv_path: CSV文件路径
        :param image_folder: 图像文件夹路径
        :param output_json_path: 输出JSON文件路径
        :param train_ratio: 训练集比例，默认为0.7
        :param val_ratio: 验证集比例，默认为0.15
        :param test_ratio: 测试集比例，默认为0.15
        """
        self.csv_path = csv_path
        self.image_folder = image_folder
        self.output_json_path = output_json_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def convert_to_json(self):
        # 读取CSV文件
        df = pd.read_csv(self.csv_path, delimiter='|')

        # 初始化数据结构
        data = {'images': []}

        # 用于跟踪当前拆分的计数
        train_count = 0
        val_count = 0
        test_count = 0

        # 根据图片名称组织数据
        for image_name, group in df.groupby('image_name'):
            # 随机确定当前图像的拆分类型
            rand_num = random.uniform(0, 1)
            if rand_num < self.train_ratio:
                split_type = 'train'
                train_count += 1
            elif rand_num < self.train_ratio + self.val_ratio:
                split_type = 'val'
                val_count += 1
            else:
                split_type = 'test'
                test_count += 1

            image_info = {
                'split': split_type,
                'filepath': os.path.join(self.image_folder, image_name),
                'filename': image_name,
                'sentences': []
            }

            # 添加每个评论的信息
            for _, row in group.iterrows():
                sentence = {
                    'tokens': row['comment'].split(),  # 将评论文本分割为单词列表
                    'raw': row['comment']
                }
                image_info['sentences'].append(sentence)

            data['images'].append(image_info)

            # 检查是否达到了所需比例，如果达到了，停止循环
            if train_count >= len(df) * self.train_ratio and val_count >= len(
                    df) * self.val_ratio and test_count >= len(df) * self.test_ratio:
                break

        # 将数据写入JSON文件
        with open(self.output_json_path, 'w') as json_file:
            json.dump(data, json_file, indent=2)

        print(f"JSON文件已创建：{self.output_json_path}")

        """
        output_json 格式参照Readme $json
        """


# 使用示例
csv_path = 'dataset/short_flickr30k_images_datasets/short_results.csv'
image_folder = 'dataset/short_flickr30k_images_datasets/short_flickr30k_images_datasets'
output_json_path = 'D:\\_01_python\\Image-Captioning\\data'

converter = DatasetConverter(csv_path, image_folder, output_json_path)
converter.convert_to_json()
