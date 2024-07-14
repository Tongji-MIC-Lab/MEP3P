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


        self.balance_bias = 0.6

        # self.depth_weight = nn.Parameter(torch.tensor(1.0))
        # self.distance_2d_weight = nn.Parameter(torch.tensor(1.0))
        # self.v_scale = nn.Parameter(torch.tensor(1.0))
        # self.balance_bias = nn.Parameter(torch.tensor(0.1))


        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, mask, depth_value, coord_3d, va_v_mask, va_a_mask, mean_depth_value=None):
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
    
        ## ====================================================================
        ## add depth information
        bias_matrix = torch.zeros([q.shape[0], 8, q.shape[2], q.shape[2]]).cuda()
        ## [b,len]->[b,len,1]
        depth_value = depth_value.unsqueeze(-1)
        ## [b,len,1]->[b,1,len]
        trans_depth_value = depth_value.transpose(2,1)
        ## depth difference->[b,len,len]
        depth_diff = abs(depth_value - trans_depth_value)
        ## [b,len,len]->[b*4,len,len]
        depth_diff = depth_diff.unsqueeze(1).repeat(1,4,1,1).reshape(depth_diff.shape[0]*4, depth_diff.shape[1], depth_diff.shape[2])
        ## [b*4,len,len]->[b*4,8,len,len]
        depth_diff = depth_diff.unsqueeze(1).repeat(1,8,1,1)
        ## mask self
        self_mask = 1.0 - torch.eye(va_v_mask.shape[1]).cuda().unsqueeze(0).unsqueeze(0).repeat(depth_diff.shape[0],8,1,1)

        depth_term = abs(x[:, :, :va_v_mask.shape[1], :va_v_mask.shape[1]])*(torch.exp(-depth_diff)-self.balance_bias)*self_mask

        bias_matrix[:, :, :va_v_mask.shape[1], :va_v_mask.shape[1]] = depth_term

        x_av = x[:, :, va_v_mask.shape[1]:, :va_v_mask.shape[1]]
        att_score_av = x_av.masked_fill((mask[:, :, va_v_mask.shape[1]:, :va_v_mask.shape[1]].int()).to(torch.bool)==False, -1e9)
        att_map_av = masked_softmax(att_score_av, mask[:, :, va_v_mask.shape[1]:, :va_v_mask.shape[1]], dim=3)

        if mean_depth_value is not None:
            # [b]->[b, a_len, 1]
            mean_depth_value = mean_depth_value.unsqueeze(-1)
            # [b, a_len, o_1en]
            trans_depth_value = trans_depth_value.unsqueeze(1).repeat(1,4,1,1).reshape(trans_depth_value.shape[0]*4, trans_depth_value.shape[1], trans_depth_value.shape[2])
            depth_diff_va = abs(mean_depth_value.repeat(1, 1 , va_v_mask.shape[1]) - trans_depth_value.repeat(1, va_a_mask.shape[1], 1))

            #depth_diff_va = depth_diff_va.unsqueeze(1).repeat(1,4,1,1).reshape(depth_diff_va.shape[0]*4, depth_diff_va.shape[1], depth_diff_va.shape[2])
            depth_diff_va = depth_diff_va.unsqueeze(1).repeat(1,8,1,1)

            #trans_depth_diff_va = depth_diff_va.transpose(2,3)

            depth_av_term = abs(x[:, :, va_v_mask.shape[1]:, :va_v_mask.shape[1]])*(torch.exp(-depth_diff_va)-self.balance_bias)
            
            
            bias_matrix[:, :, va_v_mask.shape[1]:, :va_v_mask.shape[1]] = depth_av_term

        x = x + bias_matrix
        #print("reviced:", att_score[0,0,0:4,:])
        ## ====================================================================

        att_score = x.masked_fill((mask.int()).to(torch.bool)==False, -1e9)

        att_map = masked_softmax(att_score, mask, dim=3)
        
        att_map = self.att_dropout(att_map)
        
        x = att_map.matmul(v)  # [b, h, q_len, attn]

        # [b,len,1]->[b,1,len,1]
        tmp_depth_value = depth_value.unsqueeze(1).repeat(1,4,1,1).reshape(depth_value.shape[0]*4, depth_value.shape[1], depth_value.shape[2])
        # [b, 8, o_len, 1]
        tmp_depth_value = tmp_depth_value.unsqueeze(1).repeat(1,8,1,1)
        # [b, 8, a_len, 1]->[b,a_len]
        mean_depth_value = att_map_av.matmul(tmp_depth_value).squeeze(-1)
        # [b,a_len]
        mean_depth_value = torch.mean(mean_depth_value, dim=1)

        mean_depth_value = (torch.sum(mean_depth_value, dim=-1)/torch.sum(va_a_mask.squeeze(-1), dim=-1)).unsqueeze(1).repeat(1,va_a_mask.shape[1])
        #print(mean_depth_value)
        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)
        
        x = self.output_layer(x)  
        
        assert x.size() == orig_q_size
        return x, att_score, att_map, mean_depth_value


class DepthEncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, out_size, dropout_rate, attention_dropout_rate, head_size):
        super(DepthEncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, out_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, va_mask, depth_value, coord_3d, va_v_mask, va_a_mask, mean_depth_value = None):

        y = self.self_attention_norm(x)
        y, att_score, att_map, mean_depth_value = self.self_attention(y, y, y, va_mask, depth_value, coord_3d, va_v_mask, va_a_mask, mean_depth_value)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)     
        y = self.ffn_dropout(y)
        x = x + y
        return x, att_score, att_map, mean_depth_value