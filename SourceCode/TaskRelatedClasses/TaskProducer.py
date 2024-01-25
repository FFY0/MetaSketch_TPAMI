import numpy as np
import torch

from AbstractClass.TaskRelatedClasses import AbstractTaskProducer, AbstractMetaTask, SupportGeneratorInterface, \
    AbstractQueryGenerator
from SourceCode.TaskRelatedClasses.TaskData import MetaTask, SupportSet, QuerySet

def generate_meta_task_once(taskProducer):
    taskProducer.support_generator.flush_tensor()
    support_x, support_y, _ = taskProducer.support_generator.sample_train_support()
    query_x, query_y = taskProducer.query_generator.generate_train_query(support_x, support_y, _)
    return support_x,support_y,query_x,query_y

def check_support_y_query_y(support_y, query_y):
    check_stream_length = query_y[:, 0]
    sum_stream_length = check_stream_length.sum().item()
    steam_length = support_y.sum()
    if steam_length != sum_stream_length:
        print('error!')


class TaskProducer(AbstractTaskProducer):
    def __init__(self, support_generator: SupportGeneratorInterface, query_generator: AbstractQueryGenerator,
                 device, test_task_item_size_list, test_task_group_size, zipf_decorate=False,
                 test_zipf_param_list=None,early_stop = False):
        super(TaskProducer, self).__init__(support_generator, query_generator)
        self.support_generator = support_generator
        self.query_generator = query_generator
        self.device = device
        self.test_task_item_size_list = test_task_item_size_list
        self.test_task_group_size = test_task_group_size
        self.zipf_decorate = zipf_decorate
        self.test_zipf_param_list = test_zipf_param_list
        if self.zipf_decorate:
            if self.test_zipf_param_list is None:
                print('error! Producer need zipf_param_list')
                exit()
        self.early_stop = early_stop

    def produce_train_task(self, q, pass_cuda_tensor):
        self.support_generator.flush_tensor()
        while True:
            support_x, support_y, _ = self.support_generator.sample_train_support()
            query_x, query_y = self.query_generator.generate_train_query(support_x, support_y, _)

            if not pass_cuda_tensor:
                support_x = support_x.cpu()
                support_y = support_y.cpu()
                query_x = query_x.cpu()
                query_y = query_y.cpu()

            # multi-process synchronization
            q.put(support_x)
            q.put(support_y)
            q.put(query_x)
            q.put(query_y)

    def produce_test_task(self, ):
        meta_task_group_discribe_list = []
        test_meta_task_group_list = []
        sample_support_func = None
        if self.early_stop is False:
            sample_support_func = self.support_generator.sample_test_support
        else:
            sample_support_func = self.support_generator.sample_train_support

        for test_item_size in self.test_task_item_size_list:
            print('_' + str(test_item_size) + '_itemsize_')
            meta_task_group_discribe_list.append('_' + str(test_item_size) + '_itemsize_')
            task_list = []
            for i in range(self.test_task_group_size):
                test_support_x, test_support_y, _ = sample_support_func(test_item_size, skew_ratio=1)
                test_query_x, test_query_y = self.query_generator.generate_test_query(test_support_x, test_support_y, _)
                test_support_set = SupportSet(test_support_x.cpu(),
                                              test_support_y.cpu(), self.device)
                test_query_set = QuerySet(test_query_x.cpu(),
                                          test_query_y.cpu(), self.device)
                test_meta_task = MetaTask(test_support_set, test_query_set)
                task_list.append(test_meta_task)
            test_meta_task_group_list.append(task_list)
        if self.zipf_decorate is True:
            for zipf_param in self.test_zipf_param_list:
                print('zip_param' + str(zipf_param))
                for test_item_size in self.test_task_item_size_list:
                    meta_task_group_discribe_list.append(
                        str(zipf_param) + '_zipf_' + str(test_item_size) + '_itemsize_')
                    task_list = []
                    for i in range(self.test_task_group_size):
                        test_support_x, test_support_y, _ = sample_support_func(test_item_size,skew_ratio=1.0,zipf_param = zipf_param)
                        test_query_x, test_query_y = self.query_generator.generate_test_query(test_support_x,
                                                                                              test_support_y, _)
                        test_support_set = SupportSet(test_support_x.cpu(),
                                                      test_support_y.cpu(), self.device)
                        test_query_set = QuerySet(test_query_x.cpu(),
                                                  test_query_y.cpu(), self.device)
                        test_meta_task = MetaTask(test_support_set, test_query_set)
                        task_list.append(test_meta_task)
                    test_meta_task_group_list.append(task_list)
            self.support_generator.decorate_test = False
        return test_meta_task_group_list, meta_task_group_discribe_list
