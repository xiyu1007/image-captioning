import os
from colorama import init, Fore
# 初始化 colorama
init(autoreset=True)


class PathChecker:
    def __init__(self, current_directory=None):
        """
        param current_directory: 以Image-Captioning为根目录
        """
        self.current_directory = current_directory or os.getcwd()

    def check_path_exists(self, path, create_if_not_exist=False):
        full_path = os.path.join(self.current_directory, path)

        # 检查目录是否存在
        if os.path.exists(full_path):
            return True
        else:
            print(Fore.RED + f"The directory or file '{full_path}' does not exist.")

            # 如果设置了create_if_not_exist为True，自动创建目录
            if create_if_not_exist:
                os.makedirs(full_path)
                print(Fore.BLUE + f"Directory '{full_path}' created.")
                return True  # 返回True表示目录被创建
            else:
                return False  # 返回False表示目录不存在

    def check_and_create_file(self, file_path, file_format='txt'):
        full_path = os.path.join(self.current_directory, file_path).replace("\\", "/")
        parent_path = os.path.dirname(full_path)
        temp_path = full_path
        extension = f'.{file_format.lower()}'

        count = 1
        if self.check_path_exists(parent_path, True):
            if not os.path.exists(f"{full_path}{extension}"):
                full_path = full_path + extension
            else:
                while os.path.exists(temp_path + "_" + str(count) + extension):
                    count += 1
                full_path = temp_path + "_" + str(count) + extension

            try:
                with open(full_path, 'w'):
                    # Create the file
                    return full_path
            except Exception as e:
                print(Fore.RED + f"Error while creating JSON file: {e}")
        else:
            return None
