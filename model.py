import math

from transformers.models.bert.modeling_bert import BertModel,BertPreTrainedModel
import torch.nn as nn
import torch
#from util import AxialAttention
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
            attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads)))   #调用PermuteToFrom、SelfAttention、permutation函数

        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def calculate_permutations(self,num_dimensions, emb_dim):
        total_dimensions = num_dimensions + 2
        emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
        axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]     #list

        permutations = []

        for axial_dim in axial_dims:
            last_two_dims = [axial_dim, emb_dim]       #--------list
            dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
            permutation = [*dims_rest, *last_two_dims]
            permutations.append(permutation)         #-----------append函数

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
        self.attn1 = AxialAttention(                              #调用AxialAttention函数
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
        super(TIEB, self).__init__(config)  #子类的构造函数中调用父类的构造函数，以便完成对父类属性和方法的继承，以及执行父类所需的初始化操作。
        self.bert=BertModel(config=config)
        #参数 config.fix_bert_embeddings 可能是用来控制是否固定BERT模型的词嵌入参数的一个标志
        if config.fix_bert_embeddings:
            self.bert.embeddings.word_embeddings.weight.requires_grad = False  #那么意味着这些参数在反向传播时会计算梯度，并且在参数更新时会被修改。相反，如果返回False，则意味着这些参数在反向传播时不会计算梯度，通常是因为它们被固定（即不可训练）
            self.bert.embeddings.position_embeddings.weight.requires_grad = False   #位置嵌入是否需要计算梯度
            self.bert.embeddings.token_type_embeddings.weight.requires_grad = False   #...

        #self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.dropout = nn.Dropout(config.dropout_prob)  # 用于防止过拟合
        self.dropout_2 = nn.Dropout(config.entity_pair_dropout)
        self.activation = nn.Tanh()  # 激活函数
        self.AxialAttention=BiAxialAttention(2,config.hidden_size)
        self.W = nn.Linear(3 * config.hidden_size, config.hidden_size)
        #self.relation_matrix = nn.Linear(config.hidden_size * 3, config.hidden_size)
        #self.projection_matrix = nn.Linear(config.hidden_size * 2, config.hidden_size)  # 将输入的特征向量进行线性变换，以增加特征的维度
        self.Cr = nn.Linear(config.hidden_size * 3, config.num_p * config.num_label)
        torch.nn.init.orthogonal_(self.Cr.weight, gain=1)
        #add
        self.FNN=nn.Linear(config.hidden_size * 2,config.hidden_size)
        self.H=nn.Linear(config.hidden_size * 2,config.hidden_size )
        self.FNN_3=nn.Linear(config.hidden_size*2,config.num_p * config.num_label)
    '''
    def rutn(self,x,y):
        xmat = x
        x, y = torch.broadcast_tensors(x[:, :, None], y[:, None, :])  #[b,l,h]转换成[b,l,l,h]
        max_len = xmat.shape[1]  # seq_len
        xmat_t = xmat.transpose(1, 2)  # 行列转置  [batch_size,embedding_size,seq_len]
        batch_size = xmat.shape[0]
        context = torch.ones_like(x).to('cuda')  # 生成与x形状相同、元素全为1的张量[b,l,l,h]
        for i in range(max_len):
            diag = x.diagonal(dim1=1, dim2=2, offset=-i)  # 用于返回对角线元素
            xmat_t = torch.max(xmat_t[:, :, :max_len - i], diag)
            bb = [[b] for b in range(batch_size)]
            linexup = [[j for j in range(max_len - i)] ]
            lineyup = [[j + i for j in range(max_len - i)] ]
            linexdown = [[j + i for j in range(max_len - i)] ]
            lineydown = [[j for j in range(max_len - i)] ]
            context[bb, linexup, lineyup, :] = xmat_t.permute(0, 2, 1)
            context[bb, linexdown, lineydown, :] = xmat_t.permute(0, 2, 1)
        #这个迭代不改变context的结构，[b,l,l,h]
        t = torch.cat([x, y, context], dim=-1)  # [b,l,l,h*3]
        t = self.W(t)  #[b,l,l,h]
        return t
        '''
    def forward(self, token_ids, mask_token_ids):
        embed=self.get_embed(token_ids, mask_token_ids)    #调用了get_embed函数，tensor(batch_size,seq_len,embedding_len)
        batch_size=embed.shape[0]
        seq_len = embed.shape[1]  # seq_len
        #add

        x, y = torch.broadcast_tensors(embed[:, :, None], embed[:, None, :])
        t = torch.cat([x, y], dim=-1)  # [b,l,l,h*2]
        entity_pairs = self.H(t)    #[b,l,l,h]



        #entity_pairs=self.rutn(embed,embed) #[b,l,l,h]

        #add

        distance = torch.ones_like(entity_pairs).to('cuda')        #ablation1
        for i in range(1, seq_len):
            for j in range(1, seq_len):
                distance[:, i, j, :] = math.sqrt(((i - j + 1e-2) / seq_len) ** 2)
  
        entity_pairs=torch.cat((entity_pairs,distance),dim=-1)
        


        entity_pairs = self.dropout_2(entity_pairs)         #ablation1
        entity_pairs = self.activation(entity_pairs)

        #add
        entity_pairs=self.FNN(entity_pairs)        #ablation1      ablation2更改了输出维度和名称
        
        #注释掉
        #entity_pairs = entity_pairs.reshape(batch_size, seq_len, seq_len, bert_dim)  #[batch_size, seq_len , seq_len, bert_dim(768)]
        output = self.AxialAttention(entity_pairs)  ##[batch_size, seq_len , seq_len, bert_dim(768)*3]  #ablation2没改
        table = self.Cr(output)  #----config.num_p * config.num_label              #ablation2  没改

        #table=self.FNN_3(t)
        return table.reshape([batch_size,seq_len,seq_len,self.config.num_p,self.config.num_label])
        #reshape()函数用于在不更改数据的情况下为数组赋予新形状

    def get_embed(self,token_ids, mask_token_ids):
        bert_out = self.bert(input_ids=token_ids.long(), attention_mask=mask_token_ids.long())
        embed=bert_out[0]
        embed=self.dropout(embed)   #最大池化
        return embed

    '''
            head_representation = embed.unsqueeze(2).expand(batch_size, seq_len, seq_len, bert_dim).reshape(batch_size, seq_len * seq_len, bert_dim) #unsqueeze增加一个第三维度，然后expang将第三维度扩展l倍，再重构形状[b,seq*seq,h]
            tail_representation = embed.repeat(1, seq_len, 1)# 相当于把第二维度复制l倍 [b,seq*seq,h]
            entity_pairs = torch.cat([head_representation, tail_representation], dim=-1)# [batch_size, seq_len * seq_len, bert_dim(768)*2]
            entity_pairs = self.projection_matrix(entity_pairs)  # [batch_size, seq_len * seq_len, bert_dim(768)]
            '''