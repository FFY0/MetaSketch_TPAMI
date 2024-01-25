import copy
import random
import time
import pandas as pd
import numpy as np
from AbstractClass.TaskRelatedClasses import AbstractTaskConsumer, AbstractMetaTask
from SourceCode.TaskRelatedClasses.TaskData import MetaTask, SupportSet, QuerySet
from pybloom_live import BloomFilter
from torch.multiprocessing import Manager, Queue, Process


def check(meta_task):
    try:
        unique_item = meta_task.support_set.support_x.shape[0]
        steam_length = meta_task.support_set.support_y.sum()
        check_stream_length = meta_task.query_set.query_y[:, 0]
        sum_stream_length = check_stream_length.sum()

        exist_num = meta_task.query_set.query_y.sum().item()
        if (steam_length != sum_stream_length or steam_length.item() == 0) and abs(exist_num-unique_item) > 0.5:
            print('error!!!!!!!!!!!!!!!!!!!!!!!!!')
    except:
        pass


class TaskConsumer(AbstractTaskConsumer):
    def __init__(self, device):
        self.device = device

    # 获取一个meta_task
    def consume_train_task(self, q, pass_cuda_tensor):
        support_x_tensor = q.get()
        support_y_tensor = q.get()
        query_x_tensor = q.get()
        query_y_tensor = q.get()

        meta_task = MetaTask(SupportSet(support_x_tensor, support_y_tensor, self.device),
                             QuerySet(query_x_tensor, query_y_tensor, self.device))
        if not pass_cuda_tensor:
            meta_task.to_device()
        # check(meta_task)
        # print('consume task size:', meta_task.support_set.support_y.shape[0])
        return meta_task

    # release shared memory
    def del_meta_task(self, meta_task):
        del meta_task.support_set.support_x
        del meta_task.support_set.support_y
        del meta_task.query_set.query_y
        del meta_task.query_set.query_x


class FakeTaskConsumer(AbstractTaskConsumer):
    def __init__(self, device):
        self.device = device
        self.task_pool = []
        self.index = 0

    def add_task_to_pool(self, meta_task):
        meta_task.to_device()
        self.task_pool.append(meta_task)


    # 获取一个meta_task
    def consume_train_task(self, q, pass_cuda_tensor):
        if self.index >= len(self.task_pool):
            self.index = 0
        return self.task_pool[self.index]

    # release shared memory
    def del_meta_task(self, meta_task):
        pass


class LKMLFakeTaskConsumer(AbstractTaskConsumer):
    def __init__(self, dataset_path, stream_length_start, stream_length_end, input_dimension, device):
        self.stream_node_vec_list = None
        self.prefix_frequency = None
        self.stream_statistics_values_list = None
        self.stream_statistics_keys_list = None
        self.stream_length_end = stream_length_end
        self.stream_length_start = stream_length_start
        self.dataset_path = dataset_path
        self.input_dimension = input_dimension
        self.rng = np.random.default_rng()
        self.preprocess()
        self.device = device
        # 随机数生成器

    # 转成numpy 和 排序
    def preprocess(self):
        lkml_df = pd.read_csv(self.dataset_path, delimiter='\t', header=None,
                              names=['from_node', 'to_node', 'weight', 'time_step'])
        # 转换成定长的input vector
        print(lkml_df.describe())
        lkml_df['from_node'] += pow(10, (self.input_dimension // 2))
        lkml_df['to_node'] += pow(10, (self.input_dimension // 2))
        stream_statistics_dic = {}
        for row in lkml_df.to_numpy():
            if str(row[0]) + ' ' + str(row[1]) in stream_statistics_dic.keys():
                stream_statistics_dic[str(row[0]) + ' ' + str(row[1])] = stream_statistics_dic[
                                                                             str(row[0]) + ' ' + str(row[1])] + row[2]
            else:
                stream_statistics_dic[str(row[0]) + ' ' + str(row[1])] = row[2]

        self.stream_statistics_keys_list = list(stream_statistics_dic.keys())
        self.stream_statistics_values_list = list(stream_statistics_dic.values())
        for i in range(len(self.stream_statistics_values_list)):
            str_list = self.stream_statistics_keys_list[i].split(" ")
            self.stream_statistics_keys_list[i] = str_list[0][1:] + str_list[1][1:]
            origin_str = self.stream_statistics_keys_list[i]
            self.stream_statistics_keys_list[i] = ""
            for j in origin_str:
                self.stream_statistics_keys_list[i] = self.stream_statistics_keys_list[i] + j + ","
            self.stream_statistics_keys_list[i] = np.fromstring(self.stream_statistics_keys_list[i], sep=',')
        self.stream_statistics_keys_list = np.array(self.stream_statistics_keys_list)
        self.stream_statistics_values_list = np.array(self.stream_statistics_values_list)
        node_vec = self.stream_statistics_keys_list.reshape((self.stream_statistics_keys_list.shape[0] * 2, -1))
        self.stream_node_vec_list = np.unique(node_vec, axis=0)
        # 排序
        sorted_index = np.argsort(self.stream_statistics_values_list)
        # 翻转index
        sorted_index = np.fliplr(np.stack((sorted_index, sorted_index)))[0]
        self.stream_statistics_keys_list = self.stream_statistics_keys_list[sorted_index]
        self.stream_statistics_values_list = self.stream_statistics_values_list[sorted_index]
        # # 生成前缀frequency准备采样
        # self.prefix_frequency = self.stream_statistics_values_list.copy()
        # for i in range(1, len(self.prefix_frequency)):
        #     self.prefix_frequency[i] += self.prefix_frequency[i - 1]

    def consume_task(self, stream_length=None) -> AbstractMetaTask:
        sampled_edge_id, sampled_edge_frequency = self.support_generate(stream_length)
        query_x, query_y = self.query_generate(sampled_edge_id, sampled_edge_frequency)
        support_set = SupportSet(sampled_edge_id, sampled_edge_frequency, self.device)
        query_set = QuerySet(query_x, query_y, self.device)
        meta_task = MetaTask(support_set, query_set)
        return meta_task

    def support_generate(self, stream_length=None):
        if stream_length is None:
            stream_length = self.rng.integers(self.stream_length_start, self.stream_length_end)
        sampled_edge_id, sampled_edge_frequency = self.calculate_and_mc_sample_task(self.stream_statistics_keys_list,
                                                                                    self.stream_statistics_values_list,
                                                                                    stream_length)
        # 进行打乱操作
        index = [i for i in range(sampled_edge_frequency.size)]
        random.shuffle(index)
        sampled_edge_frequency = sampled_edge_frequency[index]

        return sampled_edge_id, sampled_edge_frequency

    # 根据support 构建 query set，现在的策略是support set所有边都放入query set 中，同时找到等数量的不存在的边也放入queryset中
    def query_generate(self, sampled_edge_id, sampled_edge_frequency):
        positive_edge_num = sampled_edge_frequency.size
        # 通过bloom filter来快速筛选不存在边
        bloom_filter = BloomFilter(capacity=sampled_edge_id.shape[0] * 3, error_rate=0.5)
        for edge_id in sampled_edge_id:
            bloom_filter.add(edge_id.tostring())
        sampled_node_index = self.rng.integers(0, self.stream_node_vec_list.shape[0], positive_edge_num * 4)
        sampled_node_vec = self.stream_node_vec_list[sampled_node_index]
        sampled_edge_vec = sampled_node_vec.reshape((sampled_node_vec.shape[0] // 2, -1))
        positive_index = []
        for i in range(sampled_edge_vec.shape[0]):
            if not bloom_filter.add(sampled_edge_vec[i].tostring()):
                positive_index.append(i)
                if len(positive_index) >= positive_edge_num:
                    break
        # positive_edge_vec 是不存在的边的input vec
        positive_edge_vec = sampled_edge_vec[positive_index]
        query_x = np.concatenate((sampled_edge_id, positive_edge_vec))
        query_y_classification = np.ones((sampled_edge_frequency.size, 1))
        query_y_1 = np.concatenate((sampled_edge_frequency.reshape(-1, 1), query_y_classification), axis=1)
        query_y_2 = np.zeros((positive_edge_vec.shape[0], 2))
        query_y = np.concatenate((query_y_1, query_y_2))
        return query_x, query_y

    # 通过MC采样的方式 对小于1的edge 采样meta-task
    def mc_sample_for_call(self, edge_id, prefix_frequency, stream_length):
        # 生成随机数
        sample_list = self.rng.random(stream_length)
        sample_list *= prefix_frequency[-1]
        sample_list.sort()
        pos = 0
        sampled_edge_id = []
        sampled_edge_frequency = []
        # 记录是否第一次采样到
        first_flag = 0
        sample_index = 0
        while sample_index < stream_length:
            if sample_list[sample_index] < prefix_frequency[pos]:
                if first_flag == 0:
                    sampled_edge_id.append(edge_id[pos])
                    sampled_edge_frequency.append(1)
                    first_flag = 1
                    sample_index += 1
                else:
                    sampled_edge_frequency[-1] += 1
                    sample_index += 1
            else:
                pos += 1
                first_flag = 0
        return sampled_edge_id, sampled_edge_frequency

    # 通过结合直接计算和MC采样
    def calculate_and_mc_sample_task(self, edge_id, frequency, stream_length):
        edge_id_np = np.array(edge_id)
        frequency_np = np.array(frequency)
        frequency_np = frequency_np / np.sum(frequency_np) * stream_length
        index1_bigger_1 = np.where(frequency_np >= 1)
        index2_smaller_1 = np.where(frequency_np < 1)
        quick_edge_id = copy.deepcopy(edge_id_np[index1_bigger_1])
        quick_frequency = frequency_np[index1_bigger_1].round()

        if quick_frequency.size == frequency_np.size:
            return quick_edge_id, quick_frequency
        else:
            slow_edge_id = copy.deepcopy(edge_id_np[index2_smaller_1])
            slow_frequency = frequency_np[index2_smaller_1]
            # 注意这里没进行copy，节省时间
            slow_prefix_frequency = slow_frequency
            for i in range(1, slow_prefix_frequency.size):
                slow_prefix_frequency[i] += slow_prefix_frequency[i - 1]
            sampled_edge_id, sampled_edge_frequency = self.mc_sample_for_call(slow_edge_id, slow_prefix_frequency,
                                                                              stream_length - int(
                                                                                  np.sum(quick_frequency)))

            return np.concatenate((quick_edge_id, sampled_edge_id)), np.concatenate(
                (quick_frequency, sampled_edge_frequency))

    def get_test_meta_task(self, num_in_group, group_discribes):
        meta_task_group_list = []
        length = 1000
        for discribe in group_discribes:
            meta_task_group = []
            length *= 10
            for i in range(num_in_group):
                meta_task_group.append(self.consume_task(length))
            meta_task_group_list.append(meta_task_group)
        return meta_task_group_list, group_discribes

# if __name__ == '__main__':
#     lkml_fake_task_consumer = LKMLFakeTaskConsumer('../../Dataset/lkml-reply/out.lkml-reply', 100, 1000000, 10)
#     lkml_fake_task_consumer.consume_task()
