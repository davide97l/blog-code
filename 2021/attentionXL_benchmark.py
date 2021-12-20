import torch
import torch.nn as nn
from typing import Union, Optional, Dict, Callable, List
import torch.nn.functional as F
import time
import random
import numpy as np


class AttentionXL(torch.nn.Module):
    """
    Overview:
        Attention of TransformerXL.
    """
    def __init__(self, input_dim: int, head_dim: int, head_num: int, dropout: nn.Module) -> None:
        """Overview:
            Init AttentionXL.
        Arguments:
            - input_dim (:obj:`int`): dimension of input
            - head_dim (:obj:`int`): dimension of each head
            - head_num (:obj:`int`): number of heads for multi-head attention
            - dropout (:obj:`nn.Module`): dropout function
        """
        super(AttentionXL, self).__init__()
        self.head_num = head_num
        self.head_dim = head_dim
        self.dropout = dropout
        self.attention_kv = nn.Linear(input_dim, head_dim * head_num * 2)
        self.attention_q = nn.Linear(input_dim, head_dim * head_num)
        self.project = nn.Linear(head_dim * head_num, input_dim)
        self.project_pos = nn.Linear(input_dim, head_dim * head_num)  # project the positional embedding
        self.scale = 1 / (head_dim ** 0.5)

    def _rel_shift(self, x: torch.Tensor):
        """
        Overview:
            Relatively shift the attention score matrix.
        Arguments:
            - x (:obj:`torch.Tensor`): input tensor of shape (cur_seq, full_seq, bs, head_num).
        Returns:
            - x (:obj:`torch.Tensor`): input after relative shift. Shape (cur_seq, full_seq, bs, head_num).
        """
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        x = x_padded[1:].view_as(x)
        ones = torch.ones((x.size(0), x.size(1))).unsqueeze(-1).unsqueeze(-1)
        x = x * torch.tril(ones, x.size(1) - x.size(0))
        return x

    def forward_matmul(
            self,
            inputs: torch.Tensor,
            pos_embedding: torch.Tensor,
            full_input: torch.Tensor,
            u: torch.nn.Parameter,
            v: torch.nn.Parameter,
            mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        """Overview:
            Compute AttentionXL.
        Arguments:
            - inputs (:obj:`torch.Tensor`): attention input of shape (cur_seq, bs, input_dim)
            - pos_embedding (:obj:`torch.Tensor`): positional embedding of shape (full_seq, 1, full_seq)
            - full_input (:obj:`torch.Tensor`): memory + input concatenation of shape (full_seq, bs, input_dim)
            - u (:obj:`torch.nn.Parameter`): content parameter of shape (head_num, head_dim)
            - v (:obj:`torch.nn.Parameter`): position parameter of shape (head_num, head_dim)
            - mask (:obj:`Optional[torch.Tensor]`): attention mask of shape (cur_seq, full_seq, 1)
            full_seq = prev_seq + cur_seq
        Returns:
            - output (:obj:`torch.Tensor`): attention output of shape (cur_seq, bs, input_dim)
        """
        bs, cur_seq, full_seq = inputs.shape[1], inputs.shape[0], full_input.shape[0]
        prev_seq = full_seq - cur_seq

        kv = self.attention_kv(full_input)
        key, value = torch.chunk(kv, 2, dim=-1)  # full_seq x bs x num_head*dim_head
        query = self.attention_q(inputs)  # cur_seq x bs x num_head*dim_head
        r = self.project_pos(pos_embedding)  # full_seq x 1 x num_head*dim_head

        key = key.view(full_seq, bs, self.head_num, self.head_dim)
        query = query.view(cur_seq, bs, self.head_num, self.head_dim)
        value = value.view(cur_seq + prev_seq, bs, self.head_num, self.head_dim)
        r = r.view(full_seq, self.head_num, self.head_dim)

        # (query + u) * key^T
        q_u = query + u
        content_attn = q_u.permute(1, 2, 0, 3) @ key.permute(1, 2, 3, 0)  # bs x head_num x cur_seq x full_seq

        # (query + v) * R^T
        q_v = query + v
        position_attn = q_v.permute(1, 2, 0, 3) @ r.permute(1, 2, 0)  # bs x head_num x cur_seq x full_seq
        position_attn = self._rel_shift(position_attn)

        attn = content_attn + position_attn  # bs x head_num x cur_seq x full_seq
        attn.mul_(self.scale)

        # fills float('-inf') where mask is True to let softmax ignore those positions.
        if mask is not None and mask.any().item():
            mask = mask.permute(2, 0, 1).unsqueeze(1)  # 1 x 1 x cur_seq x full_seq
            assert mask.shape[2:] == attn.shape[2:]  # check shape of mask
            attn = attn.masked_fill(mask, -float("inf")).type_as(attn)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # multiply softmax output by value
        attn_vec = attn @ value.permute(1, 2, 0, 3)
        attn_vec = attn_vec.permute(2, 0, 1, 3)

        attn_vec = attn_vec.contiguous().view(cur_seq, bs, self.head_num * self.head_dim)
        # cur_seq x bs x head_num * head_dim
        output = self.dropout(self.project(attn_vec))  # cur_seq x bs x input_dim
        return output

    def forward_einsum(self,
                inputs: torch.Tensor,
                pos_embedding: torch.Tensor,
                full_input: torch.Tensor,
                u: torch.nn.Parameter,
                v: torch.nn.Parameter,
                mask: Optional[torch.Tensor] = None,
                ) -> torch.Tensor:
        """Overview:
            Compute AttentionXL.
        Arguments:
            - inputs (:obj:`torch.Tensor`): attention input of shape (cur_seq, bs, input_dim)
            - pos_embedding (:obj:`torch.Tensor`): positional embedding of shape (full_seq, 1, full_seq)
            - full_input (:obj:`torch.Tensor`): memory + input concatenation of shape (full_seq, bs, input_dim)
            - u (:obj:`torch.nn.Parameter`): content parameter of shape (head_num, head_dim)
            - v (:obj:`torch.nn.Parameter`): position parameter of shape (head_num, head_dim)
            - mask (:obj:`Optional[torch.Tensor]`): attention mask of shape (cur_seq, full_seq, 1)
            full_seq = prev_seq + cur_seq
        Returns:
            - output (:obj:`torch.Tensor`): attention output of shape (cur_seq, bs, input_dim)
        """
        bs, cur_seq, full_seq = inputs.shape[1], inputs.shape[0], full_input.shape[0]
        prev_seq = full_seq - cur_seq

        kv = self.attention_kv(full_input)
        key, value = torch.chunk(kv, 2, dim=-1)  # full_seq x bs x num_head*dim_head
        query = self.attention_q(inputs)  # cur_seq x bs x num_head*dim_head
        r = self.project_pos(pos_embedding)  # full_seq x 1 x num_head*dim_head

        # (query + u) * key^T
        content_attn = torch.einsum(
            "ibhd,jbhd->ijbh",
            (
                (query.view(cur_seq, bs, self.head_num, self.head_dim) + u),
                key.view(full_seq, bs, self.head_num, self.head_dim),
            ),
        )  # cur_seq x full_seq x bs x head_num

        # (query + v) * R^T
        position_attn = torch.einsum(
            "ibhd,jhd->ijbh",
            (
                (query.view(cur_seq, bs, self.head_num, self.head_dim) + v),
                r.view(cur_seq + prev_seq, self.head_num, self.head_dim),
            ),
        )  # cur_seq x full_seq x bs x head_num
        position_attn = self._rel_shift(position_attn)
        attn = content_attn + position_attn  # cur_seq x full_seq x bs x head_num
        attn.mul_(self.scale)

        if mask is not None and mask.any().item():
            assert mask.shape[:2] == attn.shape[:2]  # check shape of mask
            # fills float('-inf') where mask is True to let softmax ignore those positions.
            attn = attn.masked_fill(mask.unsqueeze(-1), -float("inf")).type_as(attn)
        attn = F.softmax(attn, dim=1)
        attn = self.dropout(attn)

        # multiply softmax output by value
        attn_vec = torch.einsum(
            "ijbh,jbhd->ibhd",
            (
                attn,
                value.view(cur_seq + prev_seq, bs, self.head_num, self.head_dim),
            ),
        )  # cur_seq x bs x head_num x head_dim
        attn_vec = attn_vec.contiguous().view(cur_seq, bs, self.head_num * self.head_dim)
        # cur_seq x bs x head_num * head_dim
        output = self.dropout(self.project(attn_vec))  # cur_seq x bs x input_dim
        return output


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    iterations = 10000
    dim_size = 128
    seq_len = 64
    bs = 32
    memory_len = 64
    head_num, head_dim = 2, 2
    input = torch.rand((seq_len, bs, dim_size))
    memory_input = torch.rand((seq_len+memory_len, bs, dim_size))
    pos_embedding = torch.rand(seq_len+memory_len, 1, dim_size)
    u, v = (
        torch.nn.Parameter(torch.zeros(head_num, head_dim)),
        torch.nn.Parameter(torch.zeros(head_num, head_dim)),
    )
    att = AttentionXL(dim_size, head_num, head_dim, nn.Dropout(0.))
    t0 = time.time()
    for i in range(iterations):
        a = att.forward_matmul(input, pos_embedding, memory_input, u, v, None)
    t1 = time.time()
    print('{} matmul attention took'.format(iterations), t1-t0)
    t0 = time.time()
    for i in range(iterations):
        a = att.forward_einsum(input, pos_embedding, memory_input, u, v, None)
    t1 = time.time()
    print('{} matmul attention took'.format(iterations), t1-t0)
