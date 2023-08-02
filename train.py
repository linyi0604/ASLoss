import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from dataset.dataset import get_dataset
from networks.networks import *
from utils.utils import make_report2, write_log


class Config():
    def __init__(self, config_name) -> None:
        self.config = json.load(open(config_name, 'r'))
        self.img_path = self.config['img_path']
        self.train_json_file = self.config['train_json_file']
        self.test_json_file = self.config['test_json_file']
        self.log_file = self.config['log_file']
        self.model_save_file = self.config['model_save_file']
        self.model_load_file = self.config['model_load_file']
        self.batch_size = self.config['batch_size']
        self.learning_rate = self.config['learning_rate']
        self.lambda_cls = self.config['lambda_cls']
        self.lambda_penalty = self.config['lambda_penalty']
        self.lambda_inner = self.config['lambda_inner']
        self.lambda_outer = self.config['lambda_outer']
        self.constraint = self.config['constraint']
        self.distance_loss = self.config['distance_loss']
        self.input_size = self.config['input_size']
        self.epoch = self.config['epoch']
        self.num_workers = self.config['num_workers']
        self.class_nums = self.config['class_nums']
        self.spatial_dimension = self.config['spatial_dimension']
        self.pretrain = self.config['pretrain']
        self.model_name = self.config['model_name']
        self.device = self.config['gpu_device'] if torch.cuda.is_available(
        ) else 'cpu'


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='')
    args = parser.parse_args()
    return args.config


def main():
    config_name = get_arg()
    config = Config(config_name)
    dir_paths = [
        os.path.dirname(config.log_file),
        os.path.dirname(config.model_save_file),
        os.path.dirname(config.model_load_file)
    ]

    for dir_path in dir_paths:
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)

    trainset, testset = get_dataset(config.img_path, config.train_json_file,
                                    config.test_json_file, config.input_size)

    trainloader = DataLoader(trainset,
                             config.batch_size,
                             shuffle=True,
                             num_workers=config.num_workers,
                             drop_last=False)
    testloader = DataLoader(testset,
                            config.batch_size,
                            num_workers=config.num_workers,
                            drop_last=False)

    model = eval(config.model_name)(config=config).to(config.device)

    max_test_acc = 0
    best_log = ""
    for epoch in range(config.epoch):
        ret = model.train_one_epoch(trainloader=trainloader)
        log = make_report2(ret, epoch=epoch, stage="train")

        ret = model.test_one_epoch(testloader=testloader)
        log += make_report2(ret, epoch=epoch, stage="test")
        accuracy = ret["accuracy"]
        if max_test_acc < accuracy:
            max_test_acc = accuracy
            best_log = "best until now: \n" + log + "\n"
            save_msg = model.save_state_dict()
            write_log(config.log_file, save_msg)

        write_log(config.log_file, log + best_log)

        print(log + best_log)


if __name__ == '__main__':
    main()
