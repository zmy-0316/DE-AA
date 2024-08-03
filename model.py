"""
主框架
"""

import math
from transformers.models.bert.modeling_bert import BertModel,BertPreTrainedModel
import torch.nn as nn
import torch
from operator import itemgetter


class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = self.sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def map_el_ind(self,arr, ind):
        return list(map(itemgetter(ind), arr))
    def sort_and_return_indices(self,arr):
        indices = [ind for ind in range(len(arr))]
        arr = zip(arr, indices)
        arr = sorted(arr)
        return self.map_el_ind(arr, 0), self.map_el_ind(arr, 1)

    def forward(self, x, **kwargs):
        axial = x.permute(*self.permutation).contiguous()
        shape = axial.shape
        *_, t, d = shape
        # merge all but axial dimension
        axial = axial.reshape(-1, t, d)
        # attention
        axial = self.fn(axial, **kwargs)
        # restore to original shape and permutation
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial

class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads = None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias = False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv = None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))
        b, t, d, h, e = *q.shape, self.heads, self.dim_heads
        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))
        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)
        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out

class AxialAttention(nn.Module):
    def __init__(self, dim, num_dimensions = 2, heads = 8, dim_heads = None, dim_index = -1, sum_axial_out = True):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)
        attentions = []
        for permutation in self.calculate_permutations(num_dimensions, dim_index):
            attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))  
        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def calculate_permutations(self,num_dimensions, emb_dim):
        total_dimensions = num_dimensions + 2
        emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
        axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]     
        permutations = []
        for axial_dim in axial_dims:
            last_two_dims = [axial_dim, emb_dim]    
            dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
            permutation = [*dims_rest, *last_two_dims]
            permutations.append(permutation)     
        return permutations
    def forward(self, x):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, 'input tensor does not have the correct input dimension'
        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x), self.axial_attentions))
        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out)
        return out

class BiAxialAttention(nn.Module):
    def __init__(self, num_head, dim):
        super(BiAxialAttention, self).__init__()
        self.attn1 = AxialAttention(                          
            dim=dim,  # embedding dimension
            dim_index=-1,  # where is the embedding dimension
            heads=num_head,  # number of heads for multi-head attention
            num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
        )

        self.attn2 = AxialAttention(
            dim=dim,  # embedding dimension
            dim_index=-1,  # where is the embedding dimension
            heads=num_head,  # number of heads for multi-head attention
            num_dimensions=2,  # number of axial dimensions (images is 2, video is 3, or more)
        )
    def forward(self, x):  # x:(b,l,l,h)
        x1, x2 = x, x.permute(0, 2, 1, 3)  #[BATCH,seq1,seq2,hidden_size]  [batch,seq2,seq1,hidden_size]
        x1 = self.attn1(x1)      #
        x2 = self.attn2(x2)      #
        return torch.cat((x, x1, x2), dim=-1)

class TIEB(BertPreTrainedModel):
    def __init__(self, config):
        super(TIEB, self).__init__(config)  
        self.bert = BertModel(config = config)
        if config.fix_bert_embeddings:
            self.bert.embeddings.word_embeddings.weight.requires_grad = False  
            self.bert.embeddings.position_embeddings.weight.requires_grad = False 
            self.bert.embeddings.token_type_embeddings.weight.requires_grad = False   
        #self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(config.dropout_prob)  
        self.dropout_2 = nn.Dropout(config.entity_pair_dropout)
        self.activation = nn.Tanh() 
        self.AxialAttention = BiAxialAttention(2, config.hidden_size)
        self.W = nn.Linear(3 * config.hidden_size, config.hidden_size)
        #self.relation_matrix = nn.Linear(config.hidden_size * 3, config.hidden_size)
        #self.projection_matrix = nn.Linear(config.hidden_size * 2, config.hidden_size)  
        self.Cr = nn.Linear(config.hidden_size * 3, config.num_p * config.num_label)
        torch.nn.init.orthogonal_(self.Cr.weight, gain = 1)
        #add
        self.FNN = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.H = nn.Linear(config.hidden_size * 2, config.hidden_size )
        self.FNN_3 = nn.Linear(config.hidden_size*2, config.num_p * config.num_label)

    def forward(self, token_ids, mask_token_ids):
        embed = self.get_embed(token_ids, mask_token_ids)  
        batch_size, seq_len, _ = embed.shape
        x, y = torch.broadcast_tensors(embed[:, :, None], embed[:, None, :])
        t = torch.cat([x, y], dim = -1)  # [b,l,l,h*2]
        entity_pairs = self.H(t)    #[b,l,l,h]
        #entity_pairs = self.rutn(embed, embed) #[b,l,l,h]
        i = torch.arange(seq_len).view(1, -1, 1).expand(batch_size, seq_len, seq_len).to('cuda')
        j = torch.arange(seq_len).view(1, 1, -1).expand(batch_size, seq_len, seq_len).to('cuda')
        distance = torch.sqrt(((i - j + 1e-2) / seq_len) ** 2).unsqueeze(-1).expand(-1, -1, -1, embed.shape[-1])
        entity_pairs=torch.cat((entity_pairs,distance),dim=-1)
        entity_pairs = self.dropout_2(entity_pairs)        
        entity_pairs = self.activation(entity_pairs)
        #add
        entity_pairs=self.FNN(entity_pairs)       
        #注释掉
        #entity_pairs = entity_pairs.reshape(batch_size, seq_len, seq_len, bert_dim)  #[batch_size, seq_len , seq_len, bert_dim(768)]
        output = self.AxialAttention(entity_pairs)  ##[batch_size, seq_len , seq_len, bert_dim(768) * 3] 
        table = self.Cr(output)  #----config.num_p * config.num_label           
        #table = self.FNN_3(t)
        return table.reshape([batch_size, seq_len, seq_len, self.config.num_p, self.config.num_label])
      

    def get_embed(self, token_ids, mask_token_ids):
        bert_out = self.bert(input_ids = token_ids.long(), attention_mask = mask_token_ids.long())
        embed = bert_out[0]
        embed = self.dropout(embed)   
        return embed

    
