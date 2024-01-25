import torch.multiprocessing.queue

from AbstractClass.AbstractStaticFactory import AbstractStaticFactory
from SourceCode.MetaSketch import MetaSketch
from SourceCode.ModelModule.EmbeddingModule import *
from SourceCode.ModelModule.MemoryMatrix import *
from SourceCode.ModelModule.RefineModule import *
from SourceCode.Model import *
from SourceCode.TaskRelatedClasses.QueryGenerator import  SimpleQueryGenerator
from SourceCode.TaskRelatedClasses.SupportGenerator import  ZipfWholeWordQuerySupportGenerator
from SourceCode.TaskRelatedClasses.TaskConsumer import *
from SourceCode.Logger import *
from SourceCode.TaskRelatedClasses.TaskProducer import TaskProducer
from SourceCode.ModelModule.LossFunc import *
import os
import random
import numpy as np


class Factory(AbstractStaticFactory):

    def __int__(self, config):
        super.__init__(config)
        self.seed_everything(0)

    def seed_everything(self, seed=0):
        '''
        :param seed:
        :param device:
        :return:
        '''

        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    def init_MS_by_load_model(self, model_path,prod_env):
        device = self.init_device()
        model = torch.load(model_path, map_location=torch.device('cpu'))
        print('load model done!')
        loss_func = self.init_loss_func(self.config['factory_config']['loss_class'])
        optimizer = self.init_optimizer(model)
        # model.to(device)
        task_producer, task_consumer,early_stop = self.init_producer_consumer(device)
        test_meta_task_group_list, meta_task_group_discribe_list = task_producer.produce_test_task()
        logger = self.init_logger(test_meta_task_group_list, meta_task_group_discribe_list, loss_func, prod_env,early_stop)
        MGS = MetaSketch(task_producer, task_consumer, model, loss_func, device, optimizer, logger)
        print('init MS by loading model done!')
        return MGS

    def fine_tune_MS_decode(self, model_path,prod_env):
        device = self.init_device()
        model = torch.load(model_path, map_location=torch.device('cpu'))
        print('load model done!')
        loss_func = self.init_loss_func(self.config['factory_config']['loss_class'])

        lr = self.config['train_config']['lr']
        decode_module_params_list = []
        for name, param in model.named_parameters():
            if "dec" in name:
                print('add param: ',name)
                decode_module_params_list.append(param)
        optimizer = torch.optim.Adam([
            {"params": decode_module_params_list, 'lr': lr }
        ], lr=lr)
        task_producer, task_consumer,early_stop = self.init_producer_consumer(device)
        test_meta_task_group_list, meta_task_group_discribe_list = task_producer.produce_test_task()
        logger = self.init_logger(test_meta_task_group_list, meta_task_group_discribe_list, loss_func, prod_env,early_stop)
        MGS = MetaSketch(task_producer, task_consumer, model, loss_func, device, optimizer, logger)
        print('init MS by loading model done!')
        return MGS

    def init_MGS(self, prod_env):
        print('init MGS....')
        model = self.init_model()
        loss_func = self.init_loss_func(self.config['factory_config']['loss_class'])
        optimizer = self.init_optimizer(model)
        device = self.init_device()
        # different dataset determine different producer
        task_producer, task_consumer,early_stop = self.init_producer_consumer(device)
        test_meta_task_group_list, meta_task_group_discribe_list = task_producer.produce_test_task()
        logger = self.init_logger(test_meta_task_group_list, meta_task_group_discribe_list, loss_func, prod_env,early_stop)
        MS = MetaSketch(task_producer, task_consumer, model, loss_func, device, optimizer, logger)
        print('init MS done!')
        return MS

    def init_producer_consumer(self, device):
        dataset_name = self.config["data_config"]["dataset_name"]
        input_dim = self.config["dim_config"]["input_dim"]
        skew_lower = self.config['data_config']['skew_lower']
        skew_upper = self.config['data_config']['skew_upper']
        item_lower = self.config['data_config']['item_lower']
        item_upper = self.config['data_config']['item_upper']
        dataset_path = self.config['data_config']['dataset_path']
        test_task_item_size_list = self.config['logger_config']['test_task_item_size_list']
        test_task_group_size = self.config['logger_config']['test_task_group_size']
        zipf_param_upper = self.config['data_config']['zipf_param_upper']
        zipf_param_lower = self.config['data_config']['zipf_param_lower']
        test_zipf_param_list = self.config['logger_config']['test_zipf_param_list']

        task_producer = None
        task_consumer = None
        early_stop = None
        if dataset_name == "WordQueryBasicSketch" :
            early_stop = False
            zipf_support_generator = ZipfWholeWordQuerySupportGenerator(dataset_path, input_dim=input_dim, item_lower=item_lower,
                                                               item_upper=item_upper,zipf_param_upper=zipf_param_upper,zipf_param_lower=zipf_param_lower,
                                                                        skew_lower=skew_lower,skew_upper =skew_upper)
            zipf_support_generator.set_device(device=device)
            query_generator = SimpleQueryGenerator()
            query_generator.set_device(device=device)
            # set zipf_decorate True due to zipf basic SketchCode
            task_producer = TaskProducer(zipf_support_generator, query_generator, device, test_task_item_size_list,
                                         test_task_group_size,zipf_decorate=True,test_zipf_param_list=test_zipf_param_list)

            task_consumer = TaskConsumer(device)


        return task_producer, task_consumer,early_stop

    def init_logger(self, test_meta_task_group_list, meta_task_group_discribe_list, loss_func, prod_env,early_stop):
        logger = Logger(test_meta_task_group_list, meta_task_group_discribe_list,
                        self.config['data_config']['dataset_name'],
                        config=self.config,
                        loss_func=loss_func,
                        flush_gap=self.config['logger_config']['flush_gap'],
                        train_comment=self.config['data_config']['train_comment'],
                        eval_gap=self.config['logger_config']['eval_gap'],
                        save_gap=self.config['logger_config']['save_gap'], prod_env=prod_env,early_stop=early_stop)
        return logger

    def init_device(self):
        cuda_num = self.config['train_config']['cuda_num']
        if cuda_num == -1:
            device = torch.device('cpu')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_num)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device

    def init_loss_func(self, loss_class):
        return eval(loss_class + "()")

    def init_optimizer(self, model):
        lr = self.config['train_config']['lr']
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        return optimizer

    def init_model(self):
        slot_dim = self.config["dim_config"]["slot_dim"]
        embedding_dim = self.config["dim_config"]["embedding_dim"]
        input_dim = self.config["dim_config"]["input_dim"]
        refined_dim = self.config["dim_config"]["refined_dim"]
        depth_dim = self.config["dim_config"]["depth_dim"]
        embedding_hidden_layer_size = self.config["hidden_layer_config"]["embedding_hidden_layer_size"]
        refined_hidden_layer_size = self.config["hidden_layer_config"]["refined_hidden_layer_size"]
        model = None
        if self.config['factory_config']['model_class'] == "Model":
            embedding_net = self.init_embedding_nets()
            refine_net = self.init_refine_nets()
            attention_matrix = self.init_attention_matrix(refined_dim, slot_dim, depth_dim)
            memory_matrix, decode_net = self.init_memory_matrix_and_decode_nets(slot_dim, depth_dim, embedding_dim)
            model = Model(attention_matrix, embedding_net, decode_net, refine_net, memory_matrix)

        if model is None:
            print('error! no model has been constructed')
            exit()
        return model

    def init_attention_matrix(self, refined_dim, slot_dim, depth_dim):
        return eval(self.config["factory_config"]["attention_class"] + "(refined_dim,slot_dim,depth_dim)")

    def init_embedding_nets(self):
        embedding_dim = self.config["dim_config"]["embedding_dim"]
        input_dim = self.config["dim_config"]["input_dim"]
        embedding_hidden_layer_size = self.config["hidden_layer_config"]["embedding_hidden_layer_size"]
        # construcat
        embedding_net = EmbeddingNet(input_dim, embedding_dim, embedding_hidden_layer_size)
        return embedding_net

    def init_memory_matrix_and_decode_nets(self, slot_dim, depth_dim, embedding_dim):
        memory_class = self.config["factory_config"]["memory_calss"]
        decode_class = self.config["factory_config"]["decode_weight_class"]
        decode_hidden_layer_size = self.config["hidden_layer_config"]["decode_hidden_layer_size"]

        memory_matrix = None
        read_dim = None
        # choose memory_class
        if memory_class == "BasicMemoryMatrix":
            memory_matrix = BasicMemoryMatrix(depth_dim, slot_dim, embedding_dim)
            read_dim = embedding_dim * (depth_dim + 1) + 3 * depth_dim + 1
        if memory_matrix is None:
            print("error! " + " need a correct memory_class ")
            exit()
        weight_decode_net = eval(decode_class + "(read_dim, decode_hidden_layer_size)")
        return memory_matrix, weight_decode_net

    def init_refine_nets(self):
        embedding_dim = self.config["dim_config"]["embedding_dim"]
        refined_dim = self.config["dim_config"]["refined_dim"]
        refined_hidden_layer_size = self.config["hidden_layer_config"]["refined_hidden_layer_size"]
        refine_net = RefineNet(embedding_dim, refined_dim, refined_hidden_layer_size)
        return refine_net
