import torch
import torch.nn as nn

from SourceCode.ModelModule.SparseSoftmax import Sparsemax



class AttentionMatrix(nn.Module):
    def __init__(self, refined_dim, slot_dim,depth_dim = 1):
        super().__init__()
        self.refined_dim = refined_dim
        self.slot_dim = slot_dim
        self.attention_matrix = torch.nn.Parameter(torch.rand(depth_dim,refined_dim, slot_dim,  requires_grad=True))
        self.normalize()
        self.sparse_softmax = Sparsemax(slot_dim)

    def forward(self, refined_vec):
        product_tensor = refined_vec.matmul(self.attention_matrix)
        return self.sparse_softmax(product_tensor)

    def normalize(self):
        with torch.no_grad():
            matrix_pow_2 = torch.square(self.attention_matrix)
            matrix_base = torch.sqrt(matrix_pow_2.sum(dim=1, keepdim=True))
            # automatic broadcast
            self.attention_matrix.data = self.attention_matrix.div(matrix_base)

