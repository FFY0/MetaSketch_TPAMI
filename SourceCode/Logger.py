import csv
import json
import math
import os
import random
import sys

import numpy as np
import pandas
from sklearn import metrics
import torch
import time
from AbstractClass.AbstractLogger import AbstractLogger
import shutil

class Logger(AbstractLogger):
    def __init__(self, test_meta_task_group_list, meta_task_group_discribe_list, dataset_name, config, loss_func,
                 flush_gap=10, train_comment=" ", eval_gap=1000, save_gap=10000, prod_env=False, early_stop=False):
        self.flush_gap = flush_gap
        self.config = config
        self.model_path = None
        self.save_gap = save_gap
        # self.log_path = log_path
        self.meta_task_discribe_list = meta_task_group_discribe_list
        self.test_meta_task_group_list = test_meta_task_group_list
        self.loss_func = loss_func
        # 训练注释
        self.train_comment = train_comment
        self.project_root = os.getcwd().split('ExpCode')[0]
        path = os.getcwd()
        self.path = path
        self.py_file_path = sys.argv[0]
        print(self.py_file_path)
        if prod_env:
            self.dataset_name = "Prod_" + time.strftime('%m%d_', time.localtime(time.time())) + dataset_name
        else:
            self.dataset_name = "Dev_" + time.strftime('%m%d_', time.localtime(time.time())) + dataset_name
        self.csv_writer_list = None
        self.log_file_list = None
        self.__init_log_file()
        self.file_header = None
        self.eval_gap = eval_gap
        self.early_stop = early_stop
        self.best_loss = None

    def logging(self, model, step, comment=""):
        print(step, ' step logging begin...')
        loss_list = []
        model.eval()
        for i in range(len(self.log_file_list)):
            if self.file_header is None:
                for group in self.test_meta_task_group_list:
                    for task in group:
                        task.to_device()
            log_file = self.log_file_list[i]
            csv_writer = self.csv_writer_list[i]
            test_meta_task_group = self.test_meta_task_group_list[i]
            group_test_merged_info_dict = self.eval_on_one_group(test_meta_task_group, model)

            # 如果是第一次logging，初始化所有文件的表头
            if self.file_header is None:
                self.file_header = list(group_test_merged_info_dict.keys())
                self.file_header.insert(0, "step")
                for writer in self.csv_writer_list:
                    writer.writerow(self.file_header)
            group_test_merged_info_dict['step'] = step
            row_content = []
            for key in self.file_header:
                row_content.append(group_test_merged_info_dict[key])
            loss_list.append(group_test_merged_info_dict['loss'])
            csv_writer.writerow(row_content)
            if (step // self.eval_gap) % self.flush_gap == 0:
                log_file.flush()
        model.train()
        loss = sum(loss_list)/len(loss_list)
        if self.early_stop:
            if self.best_loss is None:
                if step % self.save_gap == 1:
                    try:
                        torch.save(model, self.model_path)
                    except IOError:
                        print('save model IOError except')
                    self.best_loss = loss
            elif loss <= self.best_loss:
                    if step % self.save_gap == 1:
                        try:
                            torch.save(model, self.model_path)
                            print(step,'step early stop')
                        except IOError:
                            print('save model IOError except')
                        self.best_loss = loss
            else:
                print(step, 'step not save')
        else:
            if step % self.save_gap == 1:
                try:
                    torch.save(model, self.model_path)
                    torch.save(model,self.model_dir+str(step)+'model')
                except IOError:
                    print('save model IOError except')
        print(step, ' step logging done...')
        return

    def eval_on_one_group(self, test_meta_task_group, model):
        # store all test result for a group of meta tasks
        group_dict_list = []
        for test_meta_task in test_meta_task_group:
            mt_test_info_dict = self.eval_on_one_task(test_meta_task, model)
            group_dict_list.append(mt_test_info_dict)

        group_test_merged_info_dict = {}
        for key in group_dict_list[0].keys():
            value_mean = 0
            for dic in group_dict_list:
                value_mean += dic[key]
            value_mean /= len(group_dict_list)
            group_test_merged_info_dict[key] = value_mean
        return group_test_merged_info_dict

    def eval_on_one_task(self, test_meta_task, model):
        basic_info_dict = self.get_basic_eval_info_on_one_task(test_meta_task, model)
        additional_info_dict = self.get_additional_eval_info_on_one_task(test_meta_task, model)
        info_dict = dict(list(basic_info_dict.items()) + list(additional_info_dict.items()))
        return info_dict

    def get_sparsity(self, batch_data, dim=(1,)):
        sparsity_data = torch.where(torch.abs(batch_data - 0.0) < 0.00001, 0.0, 1.0)
        res = sparsity_data.sum(dim=dim, keepdim=False)
        return res

    # col_sparsity
    def get_additional_eval_info_on_one_task(self, test_meta_task, model):
        info_dict = {}
        with torch.no_grad():
            query_set = test_meta_task.query_set
            query_x = query_set.query_x
            query_y = query_set.query_y
            embedding = model.get_embedding(query_x)
            if embedding.sum().item() == 0:
                print('cant understand!')
            refined = model.get_refined(embedding)
            address = model.get_address(refined)

            addresses_var = (address.mean(dim=1) * address.shape[2]).var().item()
            addresses_sparsity = self.get_sparsity(address, dim=(0, 2)).mean().item()
            embedding_sparsity = self.get_sparsity(embedding, dim=1).mean().item()
            embedding_var = embedding.sum(0).var().item()
            embedding_norm = torch.sqrt((embedding.square()).sum(dim=1)).mean().item()
            embedding_l1_norm = embedding.sum(dim=-1).mean().item()
            embedding_var_normalization = (embedding / embedding.sum(dim=-1, keepdim=True)).sum(0).var().item()
            refined_var = refined.sum(0).var().item()
            refined_norm = torch.sqrt((refined.square()).sum(dim=1)).mean().item()
            refined_l1_norm = refined.sum(dim=-1).mean().item()

        info_dict['embedding_norm'] = embedding_norm
        info_dict['embedding_l1_norm'] = embedding_l1_norm
        info_dict["embedding_var"] = embedding_var
        info_dict["embedding_var_normalization"] = embedding_var_normalization
        info_dict['embedding_sparsity'] = embedding_sparsity
        info_dict['address_sparsity'] = addresses_sparsity
        info_dict['address_var'] = addresses_var
        info_dict['refined_var'] = refined_var
        info_dict['refined_norm'] = refined_norm
        info_dict['refined_l1_norm'] = refined_l1_norm
        if hasattr(model, 'attention_matrix'):
            if hasattr(model.attention_matrix, 'scale'):
                info_dict['scale'] = model.attention_matrix.scale.item()
        return info_dict

    def get_basic_eval_info_on_one_task(self, test_meta_task, model):
        info_dict = {}
        with torch.no_grad():
            support_set = test_meta_task.support_set
            query_set = test_meta_task.query_set
            stream_length = support_set.support_y.sum()
            model.clear()
            # write steam data into memory
            model.write(support_set.support_x, support_set.support_y)
            # query
            query_pred = model.query(query_set.query_x,
                                     stream_length.unsqueeze(-1).repeat(query_set.query_x.shape[0], 1))
            loss = self.loss_func(query_pred, query_set.query_y)
            query_y = query_set.query_y
            loss_sum = loss
            weight_pred = query_pred
            weight_AAE = torch.mean(torch.abs(weight_pred - query_y)).cpu().item()
            weight_ARE = torch.mean(torch.abs(weight_pred - query_y) / query_y).cpu().item()
            if math.isinf(weight_ARE):
                query_y = query_y.min()

            sorted_query_y, indices = query_y.view(-1).sort(descending=True)
            sorted_weighted_pred, indices = weight_pred.view(-1).sort(descending=True)
            index = int(sorted_query_y.size(dim=0) * 0.2)
            benchmark_weight_pred = sorted_weighted_pred[index].item()
            benchmark_query_y = sorted_query_y[index].item()
            label_query_y = torch.where(query_y > benchmark_query_y, torch.ones_like(query_y), torch.zeros_like(query_y))
            label_weight_pred = torch.where(weight_pred > benchmark_weight_pred, torch.ones_like(weight_pred),torch.zeros_like(weight_pred))
            f1 = metrics.f1_score(label_query_y.cpu(), label_weight_pred.cpu())
            weight_pred_var = weight_pred.var().cpu().item()
            weight_label_var = query_y.var().cpu().item()
            info_dict['f1_socre'] = f1
            info_dict["weight_pred_var"] = weight_pred_var
            info_dict["weight_label_var"] = weight_label_var
            info_dict['weight_ARE'] = weight_ARE
            info_dict['weight_AAE'] = weight_AAE
            info_dict['label_sum'] = query_y.sum().cpu().item()
            info_dict['pre_sum'] = weight_pred.sum().cpu().item()
            info_dict['item_num'] = weight_pred.shape[0]
            info_dict['loss'] = loss_sum.cpu().item()
        return info_dict

    def __init_log_file(self):
        self.log_file_list = []
        self.csv_writer_list = []
        time_str = time.strftime('_%m_%d_%H_%M_%S', time.localtime(time.time()))
        if not os.path.exists(os.path.join(self.project_root,
                                           'LogDir/{}/{}/'.format(self.dataset_name, self.train_comment + time_str))):
            os.makedirs(os.path.join(self.project_root,
                                     'LogDir/{}/{}/'.format(self.dataset_name, self.train_comment + time_str)))
        # info_example = ["step","embeding","col_sparsity","row_sparsity", "loss", "weight_ARE", "weight_AAE", "exist_precise", "exist_acc", "exist_F1","exist_recall"]
        self.model_path = os.path.join(self.project_root, 'LogDir/{}/{}/model'.format(self.dataset_name,
                                                                                      self.train_comment + time_str))
        self.model_dir = os.path.join(self.project_root, 'LogDir/{}/{}/'.format(self.dataset_name,
                                                                                      self.train_comment + time_str))

        config_str = json.dumps(self.config)
        config_file = open(os.path.join(self.project_root, 'LogDir/{}/{}/config'.format(self.dataset_name,
                                                                                        self.train_comment + time_str)),
                           'w',
                           newline='', encoding='utf-8')
        config_file.write(config_str)
        config_file.close()
        for meta_task_group_discribe in self.meta_task_discribe_list:
            self.log_file_list.append(open(os.path.join(self.project_root,
                                                        'LogDir/{}/{}/log{}.csv'.format(self.dataset_name,
                                                                                        self.train_comment + time_str,
                                                                                        meta_task_group_discribe)), 'w',
                                           newline='', encoding='utf-8'))
            self.csv_writer_list.append(csv.writer(self.log_file_list[-1]))
        # 这里添加一个持久化test meta task的方法
        os.makedirs(os.path.join(self.project_root,
                                 'LogDir/{}/{}/test_tasks_{}/'.format(self.dataset_name, self.train_comment + time_str,
                                                                      self.train_comment + time_str)))
        for i in range(len(self.test_meta_task_group_list)):
            os.makedirs(os.path.join(self.project_root,
                                     'LogDir/{}/{}/test_tasks_{}/{}/'.format(self.dataset_name,
                                                                             self.train_comment + time_str,
                                                                             self.train_comment + time_str,
                                                                             self.meta_task_discribe_list[i])))
            for j in range(len(self.test_meta_task_group_list[i])):
                path = os.path.join(self.project_root,
                                    'LogDir/{}/{}/test_tasks_{}/{}/{}.npz'.format(self.dataset_name,
                                                                                  self.train_comment + time_str,
                                                                                  self.train_comment + time_str,
                                                                                  self.meta_task_discribe_list[i],
                                                                                  str(j)))

                self.save_meta_task(self.test_meta_task_group_list[i][j], path)
        shutil.copytree(os.path.join(self.project_root, 'SourceCode'), os.path.join(self.project_root,
                                                                                    'LogDir/{}/{}/SourceCode/'.format(
                                                                                        self.dataset_name,
                                                                                        self.train_comment + time_str)))
        # shutil.copy(os.path.join(self.project_root,'Utils/GraphUtil.py'),os.path.join(self.project_root,
        #                          'LogDir/{}/{}/GraphUtil.py'.format(self.dataset_name,  self.train_comment+time_str)))
        # os.makedirs('LogDir/{}/{}/ExpCode{}'.format(self.dataset_name,  self.train_comment+time_str,self.py_file_path.split('ExpCode')[-1]))
        if '\\' in self.py_file_path:
            self.py_file_path = self.py_file_path.replace('\\', "/")
            self.path = self.path.replace('\\', "/")
        # print(self.py_file_path)
        os.makedirs(os.path.join(self.project_root,
                                 'LogDir/{}/{}/ExpCode/{}/'.format(self.dataset_name, self.train_comment + time_str,
                                                                   self.path.split('/')[-1])))
        shutil.copy(self.py_file_path, os.path.join(self.project_root,
                                                    'LogDir/{}/{}/ExpCode/{}/'.format(self.dataset_name,
                                                                                      self.train_comment + time_str,
                                                                                      self.path.split('/')[-1])))

    def save_meta_task(self, meta_task, path):
        np.savez(path, support_x=meta_task.support_set.support_x.cpu().numpy(),
                 support_y=meta_task.support_set.support_y.cpu().numpy(),
                 query_x=meta_task.query_set.query_x.cpu().numpy(), query_y=meta_task.query_set.query_y.cpu().numpy())

    def close_all_file(self):
        for file in self.log_file_list:
            file.close()
