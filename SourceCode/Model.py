import time

from AbstractClass.AbstractModel import *
import torch


class Model(AbstractModel):
    def __init__(self, attention_matrix, embedding_net, decode_net,
                 refine_net, memory_matrix):
        super(Model, self).__init__()
        self.refine_net = refine_net
        self.embedding_net = embedding_net
        self.memory_matrix = memory_matrix
        self.decode_net = decode_net
        self.attention_matrix = attention_matrix

    def write(self, input_x, input_y):
        embedding = self.get_embedding(input_x)
        refined = self.get_refined(embedding)
        address = self.get_address(refined)
        self.memory_matrix.write(address, embedding, input_y)

    def clear(self):
        self.memory_matrix.clear()

    def normalize_attention_matrix(self):
        self.attention_matrix.normalize()

    # !!! attention !!!
    def get_embedding(self, input_x):
        embedding = self.embedding_net(input_x)
        # embedding = embedding / embedding.sum(dim=-1,keepdim=True)
        return embedding

    def get_refined(self, embedding):
        refined = self.refine_net(embedding)
        return refined

    def get_address(self, refined):
        address = self.attention_matrix(refined)
        return address

    def query(self, input_x, stream_length):
        embedding = self.get_embedding(input_x)
        refined = self.get_refined(embedding)
        address = self.get_address(refined)
        read_info = self.memory_matrix.read(address, embedding)
        decode_info = torch.cat((read_info, embedding, stream_length), dim=1)
        weight_pred = self.decode_net(decode_info)
        return weight_pred
