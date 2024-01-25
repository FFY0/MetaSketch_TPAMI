import numpy as np
import torch
from pybloom_live import ScalableBloomFilter

from AbstractClass.TaskRelatedClasses import AbstractQueryGenerator



class SimpleQueryGenerator(AbstractQueryGenerator):
    def __init__(self):
        AbstractQueryGenerator.__init__(self)
        self.device = None

    def set_device(self, device):
        self.device = device

    def generate_train_query(self, support_x, support_y, stream_node_vec_list):
        # print(support_y.shape,"注意这个shape")
        return support_x, support_y.reshape(-1, 1)

    def generate_test_query(self, support_x, support_y, stream_node_vec_list):
        query_x = support_x
        query_y = support_y.reshape(-1, 1)
        return query_x, query_y


class QueryGeneratorForExist(AbstractQueryGenerator):
    def __init__(self, ):
        AbstractQueryGenerator.__init__(self)


    def generate_train_query(self, support_x, support_y, fake_edge_tensor):
        return self.generate_query_by_real_node(support_x, support_y, fake_edge_tensor)

    def generate_test_query(self, support_x, support_y, fake_edge_tensor):
        return self.generate_query_by_real_node(support_x, support_y, fake_edge_tensor)

    # construct query by uniform generate fake edge
    def construct_query(self,support_x,support_y,ratio = 1):
        sort,index=support_y.sort(descending=True)
        high_fre_ratio = 1
        high_index = index[:int(index.shape[0] * high_fre_ratio)]
        ratio = ratio * high_fre_ratio
        fake_edge_num = int(ratio * support_x.shape[0])
        support_x_np = support_x.cpu().numpy()
        node_nd = support_x_np.reshape(-1,support_x.shape[1]//2)
        support_x = support_x[high_index]
        support_y = support_y[high_index]


        rng = np.random.default_rng()
        sample_node_index = rng.integers(0, node_nd.shape[0], int(ratio * support_x.shape[0]//high_fre_ratio) * 4)
        sample_node = node_nd[sample_node_index]
        sample_edge =sample_node.reshape(sample_node.shape[0]//2,-1)
        edge_bf = ScalableBloomFilter(initial_capacity=node_nd.shape[0] , error_rate=0.01,
                                      mode=ScalableBloomFilter.LARGE_SET_GROWTH)
        fake_edge_list = []
        for exist_edge in support_x_np:
            edge_bf.add(exist_edge.tobytes())
        for edge in sample_edge:
            if len(fake_edge_list) == fake_edge_num:
                break
            if not edge_bf.add(edge.tobytes()):
                fake_edge_list.append(edge)
        fake_edge_nd = np.array(fake_edge_list)
        fake_edge_tensor = torch.tensor(fake_edge_nd,device=support_x.device)
        query_y = torch.cat((torch.ones(support_x.shape[0]),torch.zeros(fake_edge_tensor.shape[0])),dim=-1).float()
        query_y = query_y.to(support_x.device).view(-1,1)
        query_x = torch.cat((support_x,fake_edge_tensor))


        return query_x,query_y


    def generate_query_by_real_node(self,support_x, support_y, fake_edge_tensor):

        query_x,query_y = self.construct_query(support_x.clone(),support_y.clone())

        return query_x,query_y

