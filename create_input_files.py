import json
import os
import random
from colorama import init, Fore
import pandas as pd
from tqdm import tqdm

from utils import PathChecker
from utils import create_input_files

# 使用示例
# csv data to json
# csv among heads no space <==> image_name|comment_number|comment
csv_path = 'dataset/short_flickr30k_images_datasets/short_results.csv'
image_folder = 'dataset/short_flickr30k_images_datasets/short_flickr30k_images_datasets'
output_path_csv = 'out_data/datasets_to_json'

# 使用示例
# josn to hdf5 and json
json_path = f'out_data/datasets_to_json/short_flickr30k_images_datasets.json'
# image_folder = 'dataset/short_flickr30k_images_datasets/short_flickr30k_images_datasets'
output_path_json_hdf5 = 'out_data/img_json_hdf5'

# 初始化 colorama
init(autoreset=True)

class DatasetConverter:
    def __init__(self, csv_path, image_folder, output_path,
                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        将数据集转化为json文件，数据集格式参照Readme $data。
        :param csv_path: CSV文件路径
        :param image_folder: 图像文件夹路径
        :param output_json_path: 输出JSON文件路径(保存目录)
        :param train_ratio: 训练集比例，默认为0.7
        :param val_ratio: 验证集比例，默认为0.15
        :param test_ratio: 测试集比例，默认为0.15
        """
        self.csv_path = csv_path
        self.image_folder = image_folder
        self.json_name = os.path.basename(image_folder)
        self.output_path = output_path
        self.output_json_path = os.path.join(output_path, self.json_name)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        # 创建 PathChecker 实例
        self.path_checker = PathChecker()

        # 用于跟踪当前拆分的计数
        self.train_count = 0
        self.val_count = 0
        self.test_count = 0
        self.total_count = 0

    def convert_to_json(self, batch=0, record_parameters=True):
        # 读取CSV文件
        if batch:
            df = pd.read_csv(self.csv_path, delimiter='|', nrows=batch + 1)
        else:
            df = pd.read_csv(self.csv_path, delimiter='|')
        # 初始化数据结构
        data = {'images': []}

        # 根据图片名称组织数据
        for image_name, group in tqdm(df.groupby('image_name'), desc="CSV-Processing"):
            # 在这里执行你的操作
            # 随机确定当前图像的拆分类型
            rand_num = random.uniform(0, 1)
            if rand_num < self.train_ratio:
                split_type = 'train'
                self.train_count += 1
            elif rand_num < self.train_ratio + self.val_ratio:
                split_type = 'val'
                self.val_count += 1
            else:
                split_type = 'test'
                self.test_count += 1

            image_info = {
                'split': split_type,
                'filepath': os.path.join(self.image_folder, str(image_name)).replace("\\", "/"),
                'filename': image_name,
                'sentences': [
                    {
                        'tokens': row['comment'].rstrip('.').split(),
                        'raw': row['comment'].rstrip('.')
                    }
                    for _, row in group.iterrows()
                ]
            }

            data['images'].append(image_info)

            # # 检查是否达到了所需比例，如果达到了，停止循环
            # if train_count >= len(df) * self.train_ratio and val_count >= len(
            #         df) * self.val_ratio and test_count >= len(df) * self.test_ratio:
            #     break

        self.total_count = self.train_count + self.val_count + self.test_count
        # 打印拆分数量
        print(Fore.BLUE + f"Total count: {self.total_count}")
        print(Fore.BLUE + f"Train count: {self.train_count}")
        print(Fore.BLUE + f"Validation count: {self.val_count}")
        print(Fore.BLUE + f"Test count: {self.test_count}")

        self.output_json_path = \
            self.path_checker.check_and_create_filename(self.output_json_path, 'json')

        # 将数据写入JSON文件
        if self.output_json_path:
            try:
                with open(self.output_json_path, 'w') as json_file:
                    json.dump(data, json_file, indent=2)
                print(Fore.BLUE + f"Success: Datasets json file created ==> {self.output_json_path}")
            except Exception as e:
                print(Fore.RED + f"Error while creating JSON file: {e}")

        self.write_parameters_to_json()
        """
        output_jsonfile 格式参照Readme $json
        """

    def write_parameters_to_json(self, json_path='parameters_jsonfile'):
        json_path = os.path.join(self.output_path, json_path)
        """
        将参数写入JSON文件
        :param json_path: 输出JSON文件路径
        """
        parameters = {
            'csv_path': self.csv_path,
            'image_folder': self.image_folder,
            'output_json_path': self.output_json_path,
            'total_count': self.total_count,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'train_count': self.train_count,
            'val_count': self.val_count,
            'test_count': self.test_count,
        }
        json_path = self.path_checker.check_and_create_filename(json_path, 'json')
        try:
            with open(json_path, 'w') as json_file:
                json.dump(parameters, json_file, indent=2)
            print(Fore.BLUE + f"Success: Parameters written to JSON file ==> {json_path}")
        except Exception as e:
            print(Fore.RED + f"Error while writing parameters to JSON file: {e}")


def create_csv_to_json(csv_path, image_folder, output_path_csv):
    # csv data to json
    converter = DatasetConverter(csv_path, image_folder, output_path_csv)
    converter.convert_to_json(5000)


def check_io_file(json_path, image_folder, output_path_json_hdf5):
    # josn to hdf5 and json
    pathchecker = PathChecker()
    if pathchecker.check_path_exists(json_path):
        json_path = pathchecker.process_path(json_path)
    else:
        print(Fore.RED + f"Error: json file '{json_path}' not exists")
        return
    image_folder = pathchecker.check_path_exists(image_folder)
    output_path_json_hdf5 = pathchecker.check_path_exists(output_path_json_hdf5, True)
    return json_path, image_folder, output_path_json_hdf5


if __name__ == '__main__':

    create_csv_to_json(csv_path, image_folder, output_path_csv)

    json_path, image_folder, output_path_json_hdf5 = \
        check_io_file(json_path, image_folder, output_path_json_hdf5)
    # captions_per_image:5  min_word_freq: 5
    create_input_files('flickr30k', json_path, image_folder, 5, 5, output_path_json_hdf5, max_len=50)
