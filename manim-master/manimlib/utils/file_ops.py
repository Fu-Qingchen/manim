'''
这个文件中主要处理了文件的操作
'''

import os
import numpy as np


def add_extension_if_not_present(file_name, extension):
    '''
    如果file_name没有拓展名，则加上拓展名extension
    '''
    # This could conceivably be smarter about handling existing differing extensions
    if(file_name[-len(extension):] != extension):
        return file_name + extension
    else:
        return file_name


def guarantee_existence(path):
    '''
    返回path的绝对路径
    若path不存在，则创建
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.abspath(path)


def seek_full_path_from_defaults(file_name, default_dir, extensions):
    '''
    从默认文件夹中获取完整路径
    基本默认路径如下：
    1.当前目录下的file_name文件
    2.default_dir下(file_name + extension)
    '''
    possible_paths = [file_name]
    possible_paths += [
        os.path.join(default_dir, file_name + extension)
        for extension in ["", *extensions]
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise IOError("File {} not Found".format(file_name))


def get_sorted_integer_files(directory,
                             min_index=0,
                             max_index=np.inf,
                             remove_non_integer_files=False,
                             remove_indices_greater_than=None,
                             extension=None,
                             ):
    '''
    获取根据整数排序的文件
    在partial_movie_files的合并中用到
    '''
    indexed_files = []
    for file in os.listdir(directory):
        if '.' in file:
            index_str = file[:file.index('.')]
        else:
            index_str = file

        full_path = os.path.join(directory, file)
        if index_str.isdigit():
            index = int(index_str)
            if remove_indices_greater_than is not None:
                if index > remove_indices_greater_than:
                    os.remove(full_path)
                    continue
            if extension is not None and not file.endswith(extension):
                continue
            if index >= min_index and index < max_index:
                indexed_files.append((index, file))
        elif remove_non_integer_files:
            os.remove(full_path)
    indexed_files.sort(key=lambda p: p[0])
    return list(map(lambda p: os.path.join(directory, p[1]), indexed_files))
