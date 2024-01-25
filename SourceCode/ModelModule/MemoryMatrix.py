from torch import nn
import torch


class BasicMemoryMatrix(nn.Module):
    def __init__(self, depth_dim, slot_dim, embedding_dim):
        super().__init__()
        self.slot_dim = slot_dim
        self.depth_dim = depth_dim
        self.embedding_dim = embedding_dim
        # self.memory_matrix = nn.Parameter(torch.zeros(depth_dim,embedding_dim,slot_dim, requires_grad=False))
        self.memory_matrix = None
        self.device = None

    def clear(self):
        with torch.autograd.no_grad():
            self.memory_matrix = (torch.zeros(self.depth_dim, self.slot_dim, self.embedding_dim,device=self.device, requires_grad=True))



    # address dim : depth * batch size * slot
    def write(self, address, embedding, frequency):
        frequency_embedding = embedding * frequency.view(-1, 1)
        write_matrix = address.transpose(1, 2).matmul(frequency_embedding)
        self.memory_matrix = self.memory_matrix + write_matrix

    def read(self, address, embedding):
        read_info_tuple = self.basic_read_attention_sum(address, embedding)
        return torch.cat(read_info_tuple, dim=-1)

    # address dim : depth * batch size * slot
    def basic_read_attention_sum(self, address, embedding, read_compensate=True):
        batch_size = address.shape[1]
        basic_read_matrix = address.matmul(self.memory_matrix)
        if read_compensate:
            basic_read_matrix = (1 / address.square().sum(dim=-1, keepdim=True)) * basic_read_matrix
        # basic_read_matrix dim: batch size * depth * embedding
        cm_embedding = torch.where(embedding > 0.00001, embedding, torch.zeros_like(embedding) + 0.00001)
        zero_add_vec = torch.where(abs(embedding) < 0.0001,torch.zeros_like(embedding)+10000,torch.zeros_like(embedding))
        cm_read_info_1 = self.cm_read_1(basic_read_matrix, cm_embedding,zero_add_vec)
        cm_read_info_2 = self.cm_read_2(basic_read_matrix, cm_embedding,zero_add_vec)
        basic_read_info = basic_read_matrix.transpose(0, 1)
        basic_read_info = basic_read_info.reshape(batch_size, -1)

        return basic_read_info, cm_read_info_1, cm_read_info_2

    # noise reduction cm read head (before div all data minus smallest noise)
    def cm_read_2(self, basic_read_matrix, cm_embedding,zero_add_vec):
        min_info, _ = basic_read_matrix.min(dim=-1, keepdim=True)
        basic_read_minus_min = basic_read_matrix - min_info
        basic_read_minus_min = torch.where(abs(basic_read_minus_min)<0.0001,torch.zeros_like(basic_read_minus_min) + 100000, basic_read_minus_min)
        cm_read = (basic_read_minus_min + zero_add_vec).div(cm_embedding)
        min_cm_read, _ = torch.min(cm_read, dim=-1)
        # min_cm_read_view = min_cm_read.view(min_cm_read.shape[1],-1)
        # min_cm_read_transpose = min_cm_read.transpose(0,1)
        min_info = min_info.squeeze().transpose(0, 1)
        min_cm_read = min_cm_read.transpose(0, 1)
        return torch.cat((min_info, min_cm_read), dim=-1)

    # cm_read_head
    def cm_read_1(self, basic_read_matrix, cm_embedding,zero_add_vec):
        cm_basic_read_matrix = basic_read_matrix + zero_add_vec
        cm_read = cm_basic_read_matrix.div(cm_embedding)
        # cm_read = cm_read.view(basic_read_matrix.shape[0], -1)
        min_cm_read, _ = cm_read.min(dim=-1)
        return min_cm_read.transpose(0, 1)
