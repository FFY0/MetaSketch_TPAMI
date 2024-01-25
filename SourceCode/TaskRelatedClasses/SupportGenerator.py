import time

import torch
from AbstractClass.TaskRelatedClasses import SupportGeneratorInterface
import numpy as np
import random


class ZipfWholeWordQuerySupportGenerator(SupportGeneratorInterface):
    def __init__(self, data_path, input_dim=10,item_lower=10, item_upper=10000,zipf_param_upper=0.8, zipf_param_lower=1.3, skew_lower=1,
                 skew_upper=10):
        super().__init__()
        self.test_counts_sum = None
        self.train_counts_sum = None
        self.train_index = None
        self.test_index = None
        self.test_counts_np = None
        self.train_counts_np = None
        self.test_queries_np = None
        self.test_counts_tensor = None
        self.train_counts_tensor = None
        self.test_queries_tensor = None
        self.train_queries_tensor = None
        self.train_queries_np = None
        self.data_path = data_path
        self.input_dimension = input_dim
        self.item_lower = item_lower
        self.item_upper = item_upper
        if self.item_upper < self.item_lower:
            print('item upper should not be smaller than item_lower!')
        self.rng = np.random.default_rng()
        self.init_stastic_for_train()
        self.train_item_pos = 0
        self.test_item_pos = 0
        self.zipf_param_upper = zipf_param_upper
        self.zipf_param_lower = zipf_param_lower
        self.skew_lower = skew_lower
        self.skew_upper = skew_upper


    def set_device(self, device):
        super().set_device(device)
        self.flush_tensor()

    def flush_tensor(self):
        self.train_queries_tensor = torch.tensor(self.train_queries_np, device=self.device).float()
        self.train_counts_tensor = torch.tensor(self.train_counts_np, device=self.device).float()
        self.test_queries_tensor = torch.tensor(self.test_queries_np, device=self.device).float()
        self.test_counts_tensor = torch.tensor(self.test_counts_np, device=self.device).float()
        self.shuffle_train()
        self.shuffle_test()

    def init_stastic_for_train(self):
        train_task_file = np.load(self.data_path + "train_task_file.npz")
        test_task_file = np.load(self.data_path + "test_task_file.npz")
        self.train_queries_np = train_task_file['embeddings']
        self.train_counts_np = train_task_file['counts']
        self.test_queries_np = test_task_file['embeddings']
        self.test_counts_np = test_task_file['counts']
        print(self.test_counts_np.sum(), self.test_counts_np.sum() / self.test_counts_np.shape[0])
        self.train_counts_sum = self.train_counts_np.sum()
        self.test_counts_sum = self.test_counts_np.sum()
        self.train_index = [i for i in range(self.train_counts_np.shape[0])]
        self.test_index = [i for i in range(self.test_counts_np.shape[0])]

    def get_zipf_simple_way_zeta_compensate(self, zipf_param, size, stream_length):
        x = torch.arange(1, size + 1, device=self.device).float()
        x = x ** (-zipf_param)
        y = x / x.sum()
        labels = y * stream_length
        labels_round = labels.round()
        index = [i for i in range(labels_round.shape[0])]
        random.shuffle(index)
        labels_round = labels_round[index]
        return labels_round + 1

    def shuffle_train(self, zipf_param=None, skew_ratio=None):
        if zipf_param is None:
            zipf_param = random.random() * (self.zipf_param_upper - self.zipf_param_lower) + self.zipf_param_lower
        if skew_ratio is None:
            skew_ratio = random.random() * (self.skew_upper - self.skew_lower) + self.skew_lower
        self.train_item_pos = 0
        random.shuffle(self.train_index)
        self.train_queries_tensor = self.train_queries_tensor[self.train_index]
        self.train_counts_tensor = self.get_zipf_simple_way_zeta_compensate(zipf_param=zipf_param,
                                                                            size=self.train_queries_tensor.shape[0],
                                                                            stream_length=self.train_counts_sum * skew_ratio)

    def shuffle_test(self, zipf_param=None, skew_ratio=None):
        if skew_ratio is None:
            skew_ratio = 1
        self.test_item_pos = 0
        if zipf_param is None:
            random.shuffle(self.test_index)
            self.test_queries_tensor = torch.tensor(self.test_queries_np, device=self.device).float()
            self.test_counts_tensor = torch.tensor(self.test_counts_np, device=self.device).float()
            self.test_queries_tensor = self.test_queries_tensor[self.test_index]
            self.test_counts_tensor = self.test_counts_tensor[self.test_index] * skew_ratio
        else:
            random.shuffle(self.test_index)
            self.test_queries_tensor = self.test_queries_tensor[self.test_index]
            self.test_counts_tensor = self.get_zipf_simple_way_zeta_compensate(zipf_param=zipf_param,
                                                                               size=self.test_counts_tensor.shape[0],
                                                                               stream_length=self.test_counts_sum * skew_ratio)

    def sample_train_support(self, item_size=None, skew_ratio=None):
        if item_size is None:
            item_size = int(random.random() * (self.item_upper - self.item_lower) + self.item_lower)
        if item_size + self.train_item_pos >= self.train_counts_tensor.shape[0]:
            self.shuffle_train()
            if item_size + self.train_item_pos > self.train_counts_tensor.shape[0]:
                print('train item_size should be smaller than the number of all items')
                exit(0)

        items = self.train_queries_tensor[self.train_item_pos:item_size + self.train_item_pos]
        frequencies = self.train_counts_tensor[self.train_item_pos:item_size + self.train_item_pos]
        self.train_item_pos += item_size
        info = None
        return items.clone(), frequencies.clone(), info

    def sample_test_support(self, item_size=None, skew_ratio=None, zipf_param=None):
        if item_size is None:
            item_size = int(random.random() * (self.item_upper - self.item_lower) + self.item_lower)
        # generate preset param data
        if zipf_param is not None:
            self.shuffle_test(zipf_param=zipf_param, skew_ratio=skew_ratio)
        else:
            self.shuffle_test()
        if item_size + self.test_item_pos > self.test_counts_tensor.shape[0]:
            print('test item_size should be smaller than the number of all items')
            exit(0)
        items = self.test_queries_tensor[self.test_item_pos:item_size + self.test_item_pos]
        frequencies = self.test_counts_tensor[self.test_item_pos:item_size + self.test_item_pos]
        info = None
        self.test_item_pos += item_size
        # print(frequencies.var())
        return items.clone(), frequencies.clone(), info
