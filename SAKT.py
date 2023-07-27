"""
SSL Coding Practice
A Self-Attentive model for Knowledge Tracing
arXiv:1907.06837v1
"""

import torch
import torch.nn as nn
import numpy as np


class SAKTModel(nn.Module):
    def __init__(self, n_skill, emb_dim, num_heads, max_len, device, dropout=0.2):
        super(SAKTModel, self).__init__()
        self.n_skill = n_skill
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.max_len = max_len
        self.dropout = dropout
        self.device = device

        # network components
        self.q_embedding = nn.Embedding(self.n_skill + 1, self.emb_dim, padding_idx=n_skill)
        self.qa_embedding = nn.Embedding(self.n_skill * 2 + 2, self.emb_dim, padding_idx=n_skill*2+1)
        self.pos_embedding = nn.Embedding(self.max_len, self.emb_dim)
        self.multi_head_att = nn.MultiheadAttention(self.emb_dim, self.num_heads, self.dropout)
        self.layer_norm1 = nn.LayerNorm(self.emb_dim)
        self.layer_norm2 = nn.LayerNorm(self.emb_dim)
        self.ffn = FFN(self.emb_dim, dropout=self.dropout)
        self.pred = nn.Linear(self.emb_dim, 1, bias=True)

    def forward(self, q, qa):
        q_embed = self.q_embedding(q)  # size: (n_stu in a batch, max_len，emb_dim)
        qa_embed = self.qa_embedding(qa)
        pos_id = torch.arange(qa.size(1)).unsqueeze(0).to(self.device)
        pos_embed = self.pos_embedding(pos_id)
        qa_embed = qa_embed + pos_embed

        q_embed = q_embed.permute(1, 0, 2)  # 128*2*150
        qa_embed = qa_embed.permute(1, 0, 2)  # 128*2*150
        """
        why permute:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        """

        # mask是128*128的矩阵，上三角为True，k=1不包括对角线，True表示不参加计算
        mask = np.triu(np.ones((q_embed.size(0), q_embed.size(0))), k=1).astype("bool")
        mask = torch.from_numpy(mask).to(self.device)
        # attn_mask size(target seq len, source seq len)
        attention_out, _ = self.multi_head_att(q_embed, qa_embed, qa_embed, attn_mask=mask)
        attention_out = self.layer_norm1(attention_out + q_embed)
        attention_out = attention_out.permute(1, 0, 2)

        x = self.ffn(attention_out)
        x = self.layer_norm2(x + attention_out)
        x = self.pred(x)
        x = torch.sigmoid(x)

        return x.squeeze(-1)


# Formula (5) in paper
class FFN(nn.Module):
    def __init__(self, state_dimension, dropout):
        super(FFN, self).__init__()
        self.state_dimension = state_dimension
        self.dropout = dropout

        self.fc1 = nn.Linear(self.state_dimension, self.state_dimension)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.state_dimension, self.state_dimension)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.dropout(x)
