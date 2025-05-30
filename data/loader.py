import os.path
from os import remove
from re import split
from typing import Literal

class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def write_file(dir: str, file: str, content, op='w'):
        """
        将内容写入指定路径和文件名的文件中

        Args:
            dir: 文件路径
            file: 文件名
            content: 写入的内容
            op: 写入方式，默认为`w`，即覆盖
        """
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(f"{dir}/{file}", op) as f:
            f.writelines(content)

    @staticmethod
    def load_data_set(file: str):
        """
        加载数据集

        Args:
            file: 数据集路径
        
        Returns:
            data: List[[user, item, float(weight)], [...]]
        """
        data = []
        with open(file) as f:
            for line in f:
                # user_id, item_id, weight
                items = line.strip().split(' ')
                user_id = items[0]
                item_id = items[1]
                weight = items[2]
                data.append([user_id, item_id, weight])
        return data
