from multiprocessing import Process
import torch
from AbstractClass.AbstractMetaStructure import AbstractMetaStructure, AbstractLossFunc
from AbstractClass.AbstractModel import AbstractModel
from AbstractClass.TaskRelatedClasses import AbstractTaskConsumer


class MetaSketch(AbstractMetaStructure):

    def __init__(self, task_producer, task_consumer: AbstractTaskConsumer, model: AbstractModel,
                 loss_func: AbstractLossFunc, device,
                 optimizer, logger):
        self.task_consumer = task_consumer
        self.task_producer = task_producer
        self.device = device
        self.model = model
        # self.model.memory_matrix.device = self.device
        # self.model.to(self.device)
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.logger = logger


    def train(self, train_step, pass_cuda_tensor=False, queue_size=20,):
        # Warning!!!!!!!!!!!!!
        q = torch.multiprocessing.Queue(queue_size)
        p = Process(target=(self.task_producer).produce_train_task, args=(q, pass_cuda_tensor))
        p.start()
        self.model.memory_matrix.device = self.device
        self.model.to(self.device)
        for i in range(train_step):
            step = i + 1
            if step % self.logger.eval_gap == 1:
                self.logger.logging(self.model, step)
            print(step, ' step train  begin... ')
            meta_task = self.task_consumer.consume_train_task(q, pass_cuda_tensor)
            # print('Queue size:',q.qsize())
            support_set = meta_task.support_set
            query_set = meta_task.query_set
            with torch.no_grad():
                stream_length = support_set.support_y.sum()
            self.model.clear()
            self.model.write(support_set.support_x, support_set.support_y)
            query_pred = self.model.query(query_set.query_x,
                                          stream_length.unsqueeze(-1).repeat(query_set.query_x.shape[0], 1))
            loss = self.loss_func(query_pred, query_set.query_y)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            self.optimizer.step()
            self.optimizer.zero_grad()
            # torch.cuda.empty_cache()
            self.model.normalize_attention_matrix()
            self.task_consumer.del_meta_task(meta_task)
        p.terminate()
        self.logger.logging(self.model, train_step)
        self.logger.close_all_file()
