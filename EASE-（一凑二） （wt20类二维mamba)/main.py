import json
import argparse
from trainer import train
import logging
import torch  # 导入 PyTorch 库以处理张量类型

def main():
    """
    主函数，用于初始化参数并调用训练函数。
    """
    # 初始化命令行参数解析器并解析参数
    args = setup_parser().parse_args()
    # 从指定的配置文件中加载参数
    param = load_json(args.config)
    # 将解析的参数从Namespace转换为字典，并与从配置文件加载的参数合并
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    # 确保所有输入数据是 float 类型
    if 'data_type' not in args:
        args['data_type'] = 'float32'  # 默认设置为 float32

    # 调用训练函数并传入合并后的参数
    train(args)

def load_json(setting_path):
    """
    加载JSON配置文件。

    参数:
    setting_path (str): 配置文件的路径。

    返回:
    dict: 配置文件中的参数。
    """
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    """
    设置命令行参数解析器。

    返回:
    argparse.ArgumentParser: 初始化的参数解析器。
    """
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    # 添加一个名为'config'的参数，用于指定配置文件的路径
    parser.add_argument('--config', type=str, default='./exps/wt_mamba_64.json',
                        help='Json file of settings.')
    return parser

if __name__ == '__main__':
    main()
