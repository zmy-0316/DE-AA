from transformers import WEIGHTS_NAME,AdamW, get_linear_schedule_with_warmup
from bert4keras.tokenizers import Tokenizer                   #bert4keras是一个基于Keras和TensorFlow的BERT实现
from model import TIEB
from util import *
from tqdm import tqdm
import random
import os
import torch.nn as nn
import torch
from transformers.models.bert.modeling_bert import BertConfig
import json
import time

def search(pattern, sequence):
    n = len(pattern)    #词长
    for i in range(len(sequence)): #根据句长进行for循环
        if sequence[i:i + n] == pattern:
            return i
    return -1

def judge(ex):
    for s,_,o in ex["triple_list"]:    #提取头实体和尾实体
        if s=='' or o=='' or s not in ex["text"] or o not in ex["text"]:  #
            return False
    return True

#DataGenerator是keras中的
class data_generator(DataGenerator):
    def __init__(self,args,train_data, tokenizer,predicate_map,label_map,batch_size,random=False,is_train=True):
        super(data_generator,self).__init__(train_data,batch_size)  #子类把父类的__init__()放到自己的__init__()当中，将train_data和batch_size作为参数传递给父类的构造函数(因为类DataGenerator的_init_中需要这两个参数）
        self.max_len=args.max_len
        self.tokenizer=tokenizer
        self.predicate2id,self.id2predicate=predicate_map
        self.label2id,self.id2label=label_map
        self.random=random
        self.is_train=is_train

    def __iter__(self):
        batch_token_ids, batch_mask = [], []
        batch_label=[]
        batch_mask_label=[]
        batch_ex=[]
        for is_end, d in self.sample(self.random):      #self.sample是父类中的自定义方法


            if judge(d)==False:     #util函数
                continue
            token_ids, _ ,mask = self.tokenizer.encode(
                d['text'], max_length=self.max_len
            )           #文本语句转化为one-hot向量

            if self.is_train:
                spoes = {}  #position索引的{s:[(o,p)]}的形式：{(_,_):[(_,_,_)],(_,_):[(_,_,_),(_,_,_),(_,_,_)]........}
                for s, p, o in d['triple_list']:    #提取三元组（头实体、谓词、尾实体）
                    #self.tokenizer.encode(s)的输出：字典 input_ids就是encode的返回值, token_type_ids用于分句, attention_mask 用于掩码
                    '''tokenizer.tokenize() 返回词列表 默认首尾不加 [CLS] [SEP]
                       tokenizer.encode() 返回词id列表 默认首尾加 [CLS] [SEP]对应的词id'''
                    s = self.tokenizer.encode(s)[0][1:-1]#（掐头去尾）python中索引最后一个不取
                    p = self.predicate2id[p]   #r_index
                    o = self.tokenizer.encode(o)[0][1:-1]
                    s_idx = search(s, token_ids)        #找到头实体开始词在句子中的位置
                    o_idx = search(o, token_ids)          #找到尾实体开始词在句子中的位置
                    if s_idx != -1 and o_idx != -1:
                        s = (s_idx, s_idx + len(s) - 1)    #头实体跨度position元组
                        o = (o_idx, o_idx + len(o) - 1, p)   #尾实体跨度position+关系id构成的元组
                        if s not in spoes:
                            spoes[s] = []
                        spoes[s].append(o)


                if spoes:
                    label=np.zeros([len(token_ids), len(token_ids),len(self.id2predicate)]) #LLR           #建立|R|个表格
                    #label = ["N/A", "SMH", "SMT", "SS", "MMH", "MMT", "MSH","MST"]
                    for s in spoes:    #默认对字典的键进行操作，
                        s1,s2=s   #头实体起始——结束position
                        for o1,o2,p in spoes[s]:
                            if s1==s2 and o1==o2:
                                label[s1,o1,p]=self.label2id["S-S"]     #头实体，尾实体都是单一单词的情况，对应位置标注上标签的id
                            elif s1!=s2 and o1==o2:
                                label[s1,o1,p]=self.label2id["MB-SB"]
                                label[s2,o1,p]=self.label2id["ME-SE"]
                            elif s1==s2 and o1!=o2:
                                label[s1,o1,p]=self.label2id["SB-MB"]
                                label[s1,o2,p]=self.label2id["SE-ME"]
                            elif s1!=s2 and o1!=o2:
                                label[s1, o1,p] = self.label2id["MB-MB"]
                                label[s2, o2,p] = self.label2id["ME-ME"]

                    mask_label=np.ones(label.shape)
                    mask_label[0,:,:]=0      #标签有效范围
                    mask_label[-1,:,:]=0
                    mask_label[:,0,:]=0
                    mask_label[:,-1,:]=0

                    for a,b in zip([batch_token_ids, batch_mask,batch_label,batch_mask_label,batch_ex],
                                   [token_ids,mask,label,mask_label,d]):    #用序列解包同时遍历多个序列
                        a.append(b)     #构建batch
                    #zip()函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
                    #构建batch数量的数据存储，通过append进行添加，当数据数量等于batch_size时，进行填充等操作，并生成
                    if len(batch_token_ids) == self.batch_size or is_end:         #1==1
                        batch_token_ids, batch_mask=[sequence_padding(i) for i in [batch_token_ids, batch_mask]]
                        batch_label=mat_padding(batch_label)
                        batch_mask_label=mat_padding(batch_mask_label)
                        yield [
                            batch_token_ids, batch_mask,
                            batch_label,
                            batch_mask_label,batch_ex
                        ]
                        '''yield的函数则返回一个可迭代的 generator（生成器）对象，你可以使用for循环或者调用next()方法遍历生成器对象来提取结果'''
                        batch_token_ids, batch_mask = [], []
                        batch_label=[]
                        batch_mask_label=[]
                        batch_ex=[]

            else:
                #test: 不需要 label and label_mask
                for a, b in zip([batch_token_ids, batch_mask, batch_ex],
                                [token_ids, mask, d]):
                    a.append(b)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids, batch_mask = [sequence_padding(i) for i in [batch_token_ids, batch_mask]]
                    yield [
                        batch_token_ids, batch_mask, batch_ex
                    ]
                    batch_token_ids, batch_mask = [], []
                    batch_ex = []

def train(args):
    set_seed() #随机种子  util文件下函数

    try:
        torch.cuda.set_device(int(args.cuda_id))       #将 CUDA 设备设置为 args.cuda_id 所指定的 GPU 设备
    except:
        os.environ["CUDA_VISIBLE_DEVICES"] =args.cuda_id
    '''在实际应用中，这段代码的作用是确保 PyTorch 在进行 GPU 加速计算时使用特定的 CUDA 设备。这在多GPU环境下特别有用，可以确保模型训练或推理过程中使用指定的 GPU 资源。'''
    '''如果你有多块 GPU，并且想要控制模型和数据存储在哪一块 GPU 上，那么就需要像你之前提到的那样使用 torch.cuda.set_device 或 os.environ["CUDA_VISIBLE_DEVICES"] 来明确设置所需的 CUDA 设备。'''

    #建立路径

    output_path=os.path.join(args.file_result,args.dataset)#训练结果输出路径
    train_path=os.path.join(args.base_path,args.dataset,"train.json")
    dev_path=os.path.join(args.base_path,args.dataset,"dev.json")
    test_path=os.path.join(args.base_path,args.dataset,"test.json")
    rel2id_path=os.path.join(args.base_path,args.dataset,"rel2id.json")
    test_pred_path2 = os.path.join(output_path, "test_pred2.json")
    test_pred_path=os.path.join(output_path,"test_pred.json") #训练结束后结果索引
    dev_pred_path=os.path.join(output_path,"dev_pred.json")
    log_path=os.path.join(output_path,"log.txt")
    test_log_path=os.path.join(output_path,"test_log.txt")
    # label
    label_list=["N/A","SB-MB","SE-ME","S-S","MB-MB","ME-ME","MB-SB","ME-SE"]


    id2label,label2id={},{}
    for i,l in enumerate(label_list):
        id2label[str(i)]=l
        label2id[l]=i

    #加载数据集
    train_data = json.load(open(train_path,encoding='utf-8'))
    valid_data = json.load(open(dev_path,encoding='utf-8'))
    test_data = json.load(open(test_path,encoding='utf-8'))
    id2predicate, predicate2id = json.load(open(rel2id_path))

    tokenizer = Tokenizer(args.bert_vocab_path)
    config = BertConfig.from_pretrained(args.bert_config_path)

    config.dropout_prob = 0.1
    config.entity_pair_dropout = 0.1

    config.num_p=len(id2predicate)     #关系数量
    config.num_label=len(label_list)    #标签数量
    config.fix_bert_embeddings=args.fix_bert_embeddings

    train_model = TIEB.from_pretrained(pretrained_model_name_or_path=args.bert_model_path,config=config)
    '''通过 from_pretrained 方法加载预训练的BERT模型和配置，你的自定义模型类 TIEB 可以获得预训练模型的权重和配置信息，从而可以基于预训练模型的参数进行进一步的定制和训练。
    这样的做法通常被称为迁移学习，通过利用预训练模型的知识来加速和提升自定义模型在特定任务上的表现。'''
    train_model.to("cuda")

    if not os.path.exists(output_path):    #输出路径不存在，则进行创建
        os.makedirs(output_path)



    print_config(args)     #输出超参数   util内函数

    dataloader = data_generator(args,train_data, tokenizer,[predicate2id,id2predicate],[label2id,id2label],args.batch_size,random=True)

    dev_dataloader=data_generator(args,valid_data, tokenizer,[predicate2id,id2predicate],[label2id,id2label],args.test_batch_size,random=False,is_train=False)
    test_dataloader=data_generator(args,test_data, tokenizer,[predicate2id,id2predicate],[label2id,id2label],args.test_batch_size,random=False,is_train=False)

    t_total = len(dataloader) * args.num_train_epochs   #整个需要训练的次数，len(dataloader)是一个训练集分多少个batch，后面是训练epoch

    no_decay = ["bias", "LayerNorm.weight"]

    """ 优化器准备 """
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in train_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in train_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
    ]
    #创建一个优化器参数分组，通常用于在训练神经网络时设置不同的学习率和权重衰减（weight decay）策略
    #创建了两个参数组，分别针对需要权重衰减和不需要权重衰减的参数
    #第一个参数组包含了所有模型参数中不在no_decay列表中的参数，并将它们的权重衰减设为args.weight_decay
    #第二个参数组则包含了所有模型参数中在no_decay列表中的参数，这些参数的权重衰减被设为0.0，即不进行权重衰减
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.min_num)
    #AdamW是一种优化算法，它是对Adam优化算法的改进版本。在AdamW中，加入了权重衰减（Weight Decay）的概念
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup * t_total, num_training_steps=t_total
    )   #学习率调度器（scheduler）
    #get_linear_schedule_with_warmup 是 Hugging Face Transformers 提供的一个函数，用于创建带有预热期的线性学习率调度器。
    #num_warmup_steps 是预热期的步数
    #num_training_steps 是总的训练步数，它表示整个训练过程中的步数(epoch*batch_nums)
    t_best_f1=-1.0
    best_f1 = -1.0
    step = 0
    wait=0
    crossentropy=nn.CrossEntropyLoss(reduction="none")

    for epoch in range(args.num_train_epochs):
        train_model.train()      #训练模式
        epoch_loss = 0           #loss归零
        with tqdm(total=dataloader.__len__(), desc="train", ncols=80) as t:
            for i, batch in enumerate(dataloader):

                batch = [torch.tensor(d).to("cuda") for d in batch[:-1]]   #添加GPU标志
                batch_token_ids, batch_mask,batch_label,batch_mask_label= batch   #数据
                #time1=time.time()
                table = train_model(batch_token_ids, batch_mask) # BLLR    #模型训练

                table=table.reshape([-1,len(label_list)])  #二维：[batch*seq*seq*rel_num,num_label]将 table 重新塑造为一个二维数组，第一个维度由数据自动推断，而第二个维度的大小设为 len(label_list)，
                batch_label=batch_label.reshape([-1])   #一维：[batch*seq*seq*rel_num]将 batch_label 重新组织成一个一维的数组，其中 -1 表示该维度的大小由数据自动推断

                loss=crossentropy(table,batch_label.long())  #一维[batch*seq*seq*rel_num]
                loss=(loss*batch_mask_label.reshape([-1])).sum()

                loss.backward()   #反向传播

                step += 1
                epoch_loss += loss.item()   #loss加和    模型输出的是minibatch loss，
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.max_grad_norm)  #梯度截断防止梯度爆炸
                optimizer.step()    #通过梯度下降执行一步参数更新
                scheduler.step()  # Update learning rate schedule
                train_model.zero_grad()
                t.set_postfix(loss="%.4lf"%(loss.cpu().item()))
                t.update(1)
                #print("train time:",time.time()-time1)


        #time2=time.time()
        f1, precision, recall = evaluate(args,tokenizer,id2predicate,id2label,label2id,train_model,dev_dataloader,test_pred_path2)#测试
        #print("inference time:",time.time()-time2)

        '''
        if epoch > 10:
            t_f1, t_precision, t_recall = evaluate(args, tokenizer, id2predicate, id2label, label2id, train_model, test_dataloader,test_pred_path1)#测试
            if t_f1 > t_best_f1:
                t_best_f1 = t_f1
            with open(test_log_path, "a", encoding="utf-8") as f:
                print("epoch:%d\tt_f1:%f\tt_precision:%f\tt_recall:%f\tt_best_f1:%f\t" % (
                    int(epoch), t_f1, t_precision, t_recall, t_best_f1), file=f)# 输出测试集结果,保存到test_log.txt
        '''
        if f1 > best_f1:
            # Save model checkpoint
            best_f1 = f1
            wait=0
            torch.save(train_model.state_dict(), os.path.join(output_path, WEIGHTS_NAME))
        else:
            wait += 1
            if wait > 12:
                print('Epoch %05d: early stopping' % (epoch + 1))
                break

        epoch_loss = epoch_loss / dataloader.__len__()   #损失计算

        with open(log_path, "a", encoding="utf-8") as f:
            print("epoch:%d\tloss:%f\tf1:%f\tprecision:%f\trecall:%f\tbest_f1:%f" % (
                int(epoch), epoch_loss, f1, precision, recall, best_f1), file=f)     #保存验证集的各计算指标以及模型epoch_loss到log.txt日志中

        torch.cuda.empty_cache()
    train_model.load_state_dict(torch.load(os.path.join(output_path, WEIGHTS_NAME), map_location="cuda"))  #加载训练好的模型
    f1, precision, recall = evaluate(args,tokenizer,id2predicate,id2label,label2id,train_model,test_dataloader,test_pred_path)  #最终最优测试
    with open(log_path, "a", encoding="utf-8") as f:
        print("test： f1:%f\tprecision:%f\trecall:%f" % (f1, precision, recall), file=f)  #保存验证结果

def extract_spoes(args, tokenizer, id2predicate,id2label,label2id, model, batch_ex, batch_token_ids, batch_mask):
    #torch.nn.DataParallel 是用来并行运行模型的工具，通常用于在多个GPU上训练模型
    if isinstance(model,torch.nn.DataParallel):   #这段代码是用来检查 model 是否是 torch.nn.DataParallel 类型的实例。
        model=model.module                       #model.module 将返回包装在 DataParallel 中的模型。
    model.to("cuda")
    model.eval()

    with torch.no_grad():
        table=model(batch_token_ids, batch_mask)
        table = table.cpu().detach().numpy()#将张量分离（该张量与原始张量具有相同的值，但不再与计算图相关联）并转换为NumPy数组。
        #print(table.shape)
        #x=table[:,:,:,108,5].tolist()

        #print(x)
        #filename = "graph2.json"
        #with open(filename, 'w') as file_obj:
        #    json.dump(x, file_obj)

    def get_pred_id(table,all_tokens):

        B, L, _, R, _ = table.shape

        res = []
        for i in range(B):
            res.append([])  #二维列表

        table = table.argmax(axis=-1)  # B,L,L,R

        all_loc = np.where(table != label2id["N/A"]) #返回不包含“-”的坐标,维度：4*N(4对应上述的table结构，N就是具体的多少个非零坐标了)例如：([0 0 0], [3 4 7], [17 19 23], [108 108 105]) 是三个三元组


        res_dict = []
        for i in range(B):
            res_dict.append([])

        for i in range(len(all_loc[0])):        #遍历每个非N/A的填充结果索引坐标
            token_n=len(all_tokens[all_loc[0][i]])  #all_loc[0][i],,找到每个对应三元组属于哪个batch_size,然后all_tokens[all_loc[0][i]]找到其对应的token序列，序列长度

            if token_n-1 <= all_loc[1][i] \
                    or token_n-1<=all_loc[2][i] \
                    or 0 in [all_loc[1][i],all_loc[2][i]]:
                continue     #token序列中包含CLS和SEP特殊标识符，如果三元组中的实体组成词（也就是all_loc)的位置信息正是位于这两个地方，则不构成三元组；

            res_dict[all_loc[0][i]].append([all_loc[1][i], all_loc[2][i], all_loc[3][i]])   #维度：[batch,N,3] #往对应batch中添加三元组位置信息。例如：batch=1:[[[1, 5, 62], [1, 11, 207]]]

        for i in range(B):            #遍历每一个batch，依据标签查找三元组
            for l1, l2, r in res_dict[i]:
                if table[i, l1, l2, r] == label2id["S-S"]:
                    res[i].append([l1, l1, r, l2, l2])
                elif table[i, l1, l2, r] == label2id["SB-MB"]:
                    for l1_, l2_, r_ in res_dict[i]:
                        if r == r_ and table[i, l1_, l2_, r_] == label2id[
                            "SE-ME"] and l1_ == l1 and l2_ > l2:
                            res[i].append([l1, l1, r, l2, l2_])
                            break
                elif table[i, l1, l2, r] == label2id["MB-MB"]:
                    for l1_, l2_, r_ in res_dict[i]:
                        if r == r_ and table[i, l1_, l2_, r_] == label2id[
                            "ME-ME"] and l1_ > l1 and l2_ > l2:
                            res[i].append([l1, l1_, r, l2, l2_])
                            break
                elif table[i, l1, l2, r] == label2id["MB-SB"]:
                    for l1_, l2_, r_ in res_dict[i]:
                        if r == r_ and table[i, l1_, l2_, r_] == label2id[
                            "ME-SE"] and l1_ > l1 and l2_ == l2:
                            res[i].append([l1, l1_, r, l2, l2_])
                            break
        return res    #维度:[batch,*,5]

    all_tokens=[]
    for ex in batch_ex:
        tokens = tokenizer.tokenize(ex["text"], max_length=args.max_len)
        all_tokens.append(tokens)  #包含CLS\SEP特殊符号的tokens序列


    res_id=get_pred_id(table,all_tokens)

    batch_spo=[[] for _ in range(len(batch_ex))]    #创建空的二维列表，行数对应len(batch_ex)

    for b,ex in enumerate(batch_ex):
        text=ex["text"]
        tokens = all_tokens[b]   #有的单词会进行分割 例：['[CLS]', 'A', '.', 'C', '.', 'Ce', '##sen', '##a', ',', 'in', 'the', 'Serie', 'B', 'League', ',', 'has', '239', '##00', 'members', 'and', 'is', 'located', 'is', 'Ce', '##sen', '##a', '.', '[SEP]']
        mapping = tokenizer.rematch(text, tokens)  #tokenizer.rematch(text, tokens) 的作用是将原始文本 text 和标记化后的 tokens 进行匹配操作  例： [[], [0], [1], [2], [3], [5, 6], [7, 8, 9], [10], [12], [14, 15], [17, 18, 19], [21, 22, 23, 24, 25], [27], [29, 30, 31, 32, 33, 34], [36], [38, 39, 40], [42, 43, 44], [45, 46], [48, 49, 50, 51, 52, 53, 54], [56, 57, 58], [60, 61], [63, 64, 65, 66, 67, 68, 69], [71, 72], [74, 75], [76, 77, 78], [79], [81], []]
        for sh, st, r, oh, ot in res_id[b]:  #遍历对应batch中所有三元组

            s=(mapping[sh][0], mapping[st][-1])  #例(0, 10)
            o=(mapping[oh][0], mapping[ot][-1])

            #（s,r,o)
            batch_spo[b].append(
                (text[s[0]:s[1] + 1], id2predicate[str(r)], text[o[0]:o[1] + 1])
            )   #形如这个：('Iceland', '/location/country/capital', 'Reykjavik')   每个测试样本的抽取结果如[(),()....]

            #(s,o)
            #batch_spo[b].append((text[s[0]:s[1] + 1],text[o[0]:o[1] + 1]))
            #r
            #batch_spo[b].append(id2predicate[str(r)])
    return batch_spo


def evaluate(args,tokenizer,id2predicate,id2label,label2id,model,dataloader,evl_path):

    X, Y, Z = 1e-10, 1e-10, 1e-10  #防止计算异常
    f = open(evl_path, 'w', encoding='utf-8')    #w:文件存在则清空文件内容，不存在则创建新文件
    pbar = tqdm()
    for batch in dataloader:

        batch_ex=batch[-1]
        batch = [torch.tensor(d).to("cuda") for d in batch[:-1]]
        batch_token_ids, batch_mask = batch
        #找出对应的三元组
        batch_spo=extract_spoes(args, tokenizer, id2predicate,id2label,label2id, model, batch_ex,batch_token_ids, batch_mask)
        for i,ex in enumerate(batch_ex):     #遍历测试集样本中每个token
            R = set(batch_spo[i])#形如：{('Iceland', '/location/country/capital', 'Reykjavik'), ('Iceland', '/location/location/contains', 'Reykjavik')}
            #(s,r,o)
            T = set([(item[0], item[1], item[2]) for item in ex['triple_list']])#和R一样，只不过这里是真实标签
            #(s,o)
            #T=set([(item[0], item[2]) for item in ex['triple_list']])
            #r
            #T=set([ item[1] for item in ex['triple_list']])
            X += len(R & T)   #对应比对相等记为1
            Y += len(R)#预测三元组个数
            Z += len(T)#正确三元组个数
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            pbar.update()
            pbar.set_description(
                'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
            )
            s = json.dumps({
                'text': ex['text'],
                'triple_list': list(T),
                'triple_list_pred': list(R),
                'new': list(R - T),
                'lack': list(T - R),
            }, ensure_ascii=False, indent=4)
            f.write(s + '\n')
    pbar.close()
    f.close()
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


def test(args):
    try:
        torch.cuda.set_device(int(args.cuda_id))
    except:
        os.environ["CUDA_VISIBLE_DEVICES"] =args.cuda_id

    output_path=os.path.join(args.file_result,args.dataset)   #后面加载训练好的模型参数会用到这个路径
    #dev_path=os.path.join(args.base_path,args.dataset,"dev.json")
    test_path=os.path.join(args.base_path,args.dataset,"test.json")   #测试集路径
    rel2id_path=os.path.join(args.base_path,args.dataset,"rel2id.json")   #关系集路径
    test_pred_path = os.path.join(output_path, "test_pred_test1.json")   #输出详细测试结果的路径

    #label
    label_list = ["N/A", "SB-MB", "SE-ME", "S-S", "MB-MB", "ME-ME", "MB-SB", "ME-SE"]
    id2label,label2id={},{}
    for i,l in enumerate(label_list):
        id2label[str(i)]=l
        label2id[l]=i

    test_data = json.load(open(test_path))#加载测试集数据
    id2predicate, predicate2id = json.load(open(rel2id_path))#加载关系集
    tokenizer = Tokenizer(args.bert_vocab_path)   #初始化预训练模型
    config = BertConfig.from_pretrained(args.bert_config_path)
    config.num_p=len(id2predicate)    #关系集数量
    config.num_label=len(label_list)  #训练标签数量
    #config.rounds=args.rounds
    config.fix_bert_embeddings=args.fix_bert_embeddings
    config.entity_pair_dropout = 0.1
    config.dropout_prob = 0.1
    train_model = TIEB.from_pretrained(pretrained_model_name_or_path=args.bert_model_path,config=config)
    train_model.to("cuda")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print_config(args)

    test_dataloader = data_generator(args,test_data, tokenizer,[predicate2id,id2predicate],[label2id,id2label],args.test_batch_size,random=False,is_train=False)     #数据加载

    train_model.load_state_dict(torch.load(os.path.join(output_path, WEIGHTS_NAME), map_location="cuda"))  #加载训练好的模型
    t0=time.time()
    f1, precision, recall = evaluate(args, tokenizer, id2predicate, id2label, label2id, train_model, test_dataloader,test_pred_path)  #
    print(time.time()-t0)
    print("f1:%f,precision:%f, recall:%f"%(f1, precision, recall))
