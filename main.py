from transformers import WEIGHTS_NAME,AdamW, get_linear_schedule_with_warmup
from bert4keras.tokenizers import Tokenizer                  
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
    n = len(pattern)   
    for i in range(len(sequence)): 
        if sequence[i:i + n] == pattern:
            return i
    return -1

def judge(ex):
    for s,_,o in ex["triple_list"]:   
        if s == '' or o == '' or s not in ex["text"] or o not in ex["text"]:  
            return False
    return True


class Data_generator(DataGenerator):
    def __init__(self, args, train_data, tokenizer, predicate_map, label_map, batch_size, random = False, is_train = True):
        super(data_generator, self).__init__(train_data,batch_size) 
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.predicate2id, self.id2predicate = predicate_map
        self.label2id, self.id2label = label_map
        self.random = random
        self.is_train = is_train

    def __iter__(self):
        batch_token_ids, batch_mask = [], []
        batch_label = []
        batch_mask_label = []
        batch_ex = []
        for is_end, d in self.sample(self.random):      


            if judge(d) == False:     
                continue
            token_ids, _ ,mask = self.tokenizer.encode(
                d['text'], max_length = self.max_len
            )           

            if self.is_train:
                spoes = {}  
                for s, p, o in d['triple_list']: 
                    s = self.tokenizer.encode(s)[0][1:-1]
                    p = self.predicate2id[p]   #r_index
                    o = self.tokenizer.encode(o)[0][1:-1]
                    s_idx = search(s, token_ids)        
                    o_idx = search(o, token_ids)         
                    if s_idx != -1 and o_idx != -1:
                        s = (s_idx, s_idx + len(s) - 1)    
                        o = (o_idx, o_idx + len(o) - 1, p)  
                        if s not in spoes:
                            spoes[s] = []
                        spoes[s].append(o)


                if spoes:
                    label = np.zeros([len(token_ids), len(token_ids),len(self.id2predicate)]) #LLR           
                
                    for s in spoes:    
                        s1,s2 = s  
                        for o1,o2,p in spoes[s]:
                            if s1 == s2 and o1 == o2:
                                label[s1,o1,p]=self.label2id["S-S"]    
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
                    mask_label[0,:,:]=0    
                    mask_label[-1,:,:]=0
                    mask_label[:,0,:]=0
                    mask_label[:,-1,:]=0

                    for a,b in zip([batch_token_ids, batch_mask,batch_label,batch_mask_label,batch_ex],
                                   [token_ids,mask,label,mask_label,d]):   
                        a.append(b)    
                    
                    if len(batch_token_ids) == self.batch_size or is_end:  
                        batch_token_ids, batch_mask=[sequence_padding(i) for i in [batch_token_ids, batch_mask]]
                        batch_label=mat_padding(batch_label)
                        batch_mask_label=mat_padding(batch_mask_label)
                        yield [
                            batch_token_ids, batch_mask,
                            batch_label,
                            batch_mask_label,batch_ex
                        ]
                        
                        batch_token_ids, batch_mask = [], []
                        batch_label=[]
                        batch_mask_label=[]
                        batch_ex=[]

            else:
                
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
    set_seed()

    try:
        torch.cuda.set_device(int(args.cuda_id))      
    except:
        os.environ["CUDA_VISIBLE_DEVICES"] =args.cuda_id
   

    output_path=os.path.join(args.file_result,args.dataset)
    train_path=os.path.join(args.base_path,args.dataset,"train.json")
    dev_path=os.path.join(args.base_path,args.dataset,"dev.json")
    test_path=os.path.join(args.base_path,args.dataset,"test.json")
    rel2id_path=os.path.join(args.base_path,args.dataset,"rel2id.json")
    test_pred_path2 = os.path.join(output_path, "test_pred2.json")
    test_pred_path=os.path.join(output_path,"test_pred.json")
    dev_pred_path=os.path.join(output_path,"dev_pred.json")
    log_path=os.path.join(output_path,"log.txt")
    test_log_path=os.path.join(output_path,"test_log.txt")

   


    id2label,label2id={},{}
    for i,l in enumerate(label_list):
        id2label[str(i)]=l
        label2id[l]=i

    
    train_data = json.load(open(train_path,encoding='utf-8'))
    valid_data = json.load(open(dev_path,encoding='utf-8'))
    test_data = json.load(open(test_path,encoding='utf-8'))
    id2predicate, predicate2id = json.load(open(rel2id_path))

    tokenizer = Tokenizer(args.bert_vocab_path)
    config = BertConfig.from_pretrained(args.bert_config_path)

    config.dropout_prob = 0.1
    config.entity_pair_dropout = 0.1

    config.num_p=len(id2predicate)     
    config.num_label=len(label_list)    
    config.fix_bert_embeddings=args.fix_bert_embeddings

    train_model = TIEB.from_pretrained(pretrained_model_name_or_path=args.bert_model_path,config=config)
   
    train_model.to("cuda")

    if not os.path.exists(output_path):   
        os.makedirs(output_path)



    print_config(args)    

    dataloader = data_generator(args,train_data, tokenizer,[predicate2id,id2predicate],[label2id,id2label],args.batch_size,random=True)

    dev_dataloader=data_generator(args,valid_data, tokenizer,[predicate2id,id2predicate],[label2id,id2label],args.test_batch_size,random=False,is_train=False)
    test_dataloader=data_generator(args,test_data, tokenizer,[predicate2id,id2predicate],[label2id,id2label],args.test_batch_size,random=False,is_train=False)

    t_total = len(dataloader) * args.num_train_epochs   

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
   
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.min_num)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup * t_total, num_training_steps=t_total
    )   
    
    t_best_f1=-1.0
    best_f1 = -1.0
    step = 0
    wait=0
    crossentropy=nn.CrossEntropyLoss(reduction="none")

    for epoch in range(args.num_train_epochs):
        train_model.train()      
        epoch_loss = 0           
        with tqdm(total=dataloader.__len__(), desc="train", ncols=80) as t:
            for i, batch in enumerate(dataloader):

                batch = [torch.tensor(d).to("cuda") for d in batch[:-1]]   
                batch_token_ids, batch_mask,batch_label,batch_mask_label= batch   
                #time1=time.time()
                table = train_model(batch_token_ids, batch_mask) # BLLR    

                table=table.reshape([-1,len(label_list)])  
                batch_label=batch_label.reshape([-1])   

                loss=crossentropy(table,batch_label.long()) 
                loss=(loss*batch_mask_label.reshape([-1])).sum()

                loss.backward()   

                step += 1
                epoch_loss += loss.item()   
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), args.max_grad_norm) 
                optimizer.step()   
                scheduler.step()  
                train_model.zero_grad()
                t.set_postfix(loss="%.4lf"%(loss.cpu().item()))
                t.update(1)
                #print("train time:",time.time()-time1)


        #time2=time.time()
        f1, precision, recall = evaluate(args,tokenizer,id2predicate,id2label,label2id,train_model,dev_dataloader,test_pred_path2)
        #print("inference time:",time.time()-time2)

        '''
        if epoch > 10:
            t_f1, t_precision, t_recall = evaluate(args, tokenizer, id2predicate, id2label, label2id, train_model, test_dataloader,test_pred_path1)
            if t_f1 > t_best_f1:
                t_best_f1 = t_f1
            with open(test_log_path, "a", encoding="utf-8") as f:
                print("epoch:%d\tt_f1:%f\tt_precision:%f\tt_recall:%f\tt_best_f1:%f\t" % (
                    int(epoch), t_f1, t_precision, t_recall, t_best_f1), file=f)#
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

        epoch_loss = epoch_loss / dataloader.__len__()   

        with open(log_path, "a", encoding="utf-8") as f:
            print("epoch:%d\tloss:%f\tf1:%f\tprecision:%f\trecall:%f\tbest_f1:%f" % (
                int(epoch), epoch_loss, f1, precision, recall, best_f1), file=f)   

        torch.cuda.empty_cache()
    train_model.load_state_dict(torch.load(os.path.join(output_path, WEIGHTS_NAME), map_location="cuda")) 
    f1, precision, recall = evaluate(args,tokenizer,id2predicate,id2label,label2id,train_model,test_dataloader,test_pred_path) 
    with open(log_path, "a", encoding="utf-8") as f:
        print("test： f1:%f\tprecision:%f\trecall:%f" % (f1, precision, recall), file=f) 

def extract_spoes(args, tokenizer, id2predicate,id2label,label2id, model, batch_ex, batch_token_ids, batch_mask):
 
    if isinstance(model,torch.nn.DataParallel):   
        model=model.module                      
    model.to("cuda")
    model.eval()

    with torch.no_grad():
        table=model(batch_token_ids, batch_mask)
        table = table.cpu().detach().numpy()
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
            res.append([]) 

        table = table.argmax(axis=-1)  # B,L,L,R

        all_loc = np.where(table != label2id["N/A"]) 

        res_dict = []
        for i in range(B):
            res_dict.append([])

        for i in range(len(all_loc[0])):       
            token_n=len(all_tokens[all_loc[0][i]]) 

            if token_n-1 <= all_loc[1][i] \
                    or token_n-1<=all_loc[2][i] \
                    or 0 in [all_loc[1][i],all_loc[2][i]]:
                continue    

            res_dict[all_loc[0][i]].append([all_loc[1][i], all_loc[2][i], all_loc[3][i]])   
            
        for i in range(B):           
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
        return res    

    all_tokens=[]
    for ex in batch_ex:
        tokens = tokenizer.tokenize(ex["text"], max_length=args.max_len)
        all_tokens.append(tokens)  


    res_id=get_pred_id(table,all_tokens)

    batch_spo=[[] for _ in range(len(batch_ex))]  
    for b,ex in enumerate(batch_ex):
        text=ex["text"]
        tokens = all_tokens[b]  
        mapping = tokenizer.rematch(text, tokens) 
        for sh, st, r, oh, ot in res_id[b]: 

            s=(mapping[sh][0], mapping[st][-1])  #(0, 10)
            o=(mapping[oh][0], mapping[ot][-1])

            #（s,r,o)
            batch_spo[b].append(
                (text[s[0]:s[1] + 1], id2predicate[str(r)], text[o[0]:o[1] + 1])
            )   

            #(s,o)
            #batch_spo[b].append((text[s[0]:s[1] + 1],text[o[0]:o[1] + 1]))
            #r
            #batch_spo[b].append(id2predicate[str(r)])
    return batch_spo


def evaluate(args,tokenizer,id2predicate,id2label,label2id,model,dataloader,evl_path):

    X, Y, Z = 1e-10, 1e-10, 1e-10 
    f = open(evl_path, 'w', encoding='utf-8')   
    pbar = tqdm()
    for batch in dataloader:

        batch_ex=batch[-1]
        batch = [torch.tensor(d).to("cuda") for d in batch[:-1]]
        batch_token_ids, batch_mask = batch
    
        batch_spo=extract_spoes(args, tokenizer, id2predicate,id2label,label2id, model, batch_ex,batch_token_ids, batch_mask)
        for i,ex in enumerate(batch_ex):   
            R = set(batch_spo[i])
            #(s,r,o)
            T = set([(item[0], item[1], item[2]) for item in ex['triple_list']])
            #(s,o)
            #T=set([(item[0], item[2]) for item in ex['triple_list']])
            #r
            #T=set([ item[1] for item in ex['triple_list']])
            X += len(R & T) 
            Y += len(R)
            Z += len(T)
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

    output_path=os.path.join(args.file_result,args.dataset)  
    #dev_path=os.path.join(args.base_path,args.dataset,"dev.json")
    test_path=os.path.join(args.base_path,args.dataset,"test.json")   
    rel2id_path=os.path.join(args.base_path,args.dataset,"rel2id.json")   
    test_pred_path = os.path.join(output_path, "test_pred_test1.json")   

    #label
    label_list = ["N/A", "SB-MB", "SE-ME", "S-S", "MB-MB", "ME-ME", "MB-SB", "ME-SE"]
    id2label,label2id={},{}
    for i,l in enumerate(label_list):
        id2label[str(i)]=l
        label2id[l]=i

    test_data = json.load(open(test_path))
    id2predicate, predicate2id = json.load(open(rel2id_path))
    tokenizer = Tokenizer(args.bert_vocab_path)   
    config = BertConfig.from_pretrained(args.bert_config_path)
    config.num_p=len(id2predicate)    
    config.num_label=len(label_list)  
    #config.rounds=args.rounds
    config.fix_bert_embeddings=args.fix_bert_embeddings
    config.entity_pair_dropout = 0.1
    config.dropout_prob = 0.1
    train_model = TIEB.from_pretrained(pretrained_model_name_or_path=args.bert_model_path,config=config)
    train_model.to("cuda")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print_config(args)

    test_dataloader = data_generator(args,test_data, tokenizer,[predicate2id,id2predicate],[label2id,id2label],args.test_batch_size,random=False,is_train=False)  

    train_model.load_state_dict(torch.load(os.path.join(output_path, WEIGHTS_NAME), map_location="cuda")) 
    t0=time.time()
    f1, precision, recall = evaluate(args, tokenizer, id2predicate, id2label, label2id, train_model, test_dataloader,test_pred_path)  
    print(time.time()-t0)
    print("f1:%f,precision:%f, recall:%f"%(f1, precision, recall))
