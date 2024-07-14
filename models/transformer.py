import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from allennlp.nn.util import masked_softmax

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, out_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.layer2 = nn.Linear(ffn_size, out_size)
        self.dropout =nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, mask, att_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)
        
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)
        
        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        ## [b,len,1]->[b,h,len,1]->[b,h,len,len]
        mask = mask.unsqueeze(1).repeat(1, self.head_size, 1, 1)
        mask_trans = mask.transpose(3,2)
        mask = mask_trans*mask
        
        x = torch.matmul(q, k)* self.scale 

        if att_bias is not None:
            x = x + att_bias
        
        att_score = x.masked_fill((mask.int()).to(torch.bool)==False, -1e9)

        att_map = masked_softmax(att_score, mask, dim=3)
        
        att_map = self.att_dropout(att_map)
        
        x = att_map.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)
        
        x = self.output_layer(x)  
        
        assert x.size() == orig_q_size
        return x, att_score, att_map


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, out_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, out_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask, att_bias=None):

        y = self.self_attention_norm(x)
        y, att_score, att_map = self.self_attention(y, y, y, mask)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)     
        y = self.ffn_dropout(y)
        x = x + y
        return x, att_score, att_map