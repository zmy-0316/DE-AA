import argparse
from main import *

parser = argparse.ArgumentParser(description='Model Controller')
parser.add_argument('--dataset', default='WebNLG_star', type=str)  #['NYT','NYT_star','WebNLG','WebNLG_star']
parser.add_argument('--cuda_id', default="0", type=str)
parser.add_argument('--train', default="train", type=str)
parser.add_argument('--batch_size', default=6, type=int)   #NYT=12,NYT_star=12,WebNLG/WebNLG_star=8; #新NYT_star=12，1e-5,webnlg=8,3e-5
parser.add_argument('--test_batch_size', default=1, type=int)
parser.add_argument('--learning_rate', default=3e-5, type=float)  #1e-5=NYT/NYT_star;     WebNLG/webnlg_star=5e-5, 5e-5
parser.add_argument('--num_train_epochs', default=50, type=int)
parser.add_argument('--fix_bert_embeddings', default=False, type=bool)
parser.add_argument('--bert_vocab_path', default="Pretraining_Model/bert-base-cased/vocab.txt", type=str)
parser.add_argument('--bert_config_path', default="Pretraining_Model/bert-base-cased/config.json", type=str)
parser.add_argument('--bert_model_path', default="Pretraining_Model/bert-base-cased/pytorch_model.bin", type=str)
parser.add_argument('--max_len', default=100, type=int)
parser.add_argument('--warmup', default=0.0, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--max_grad_norm', default=1.0, type=float)
parser.add_argument('--min_num', default=1e-7, type=float)
parser.add_argument('--base_path', default="dataset", type=str)
parser.add_argument('--file_result', default="OutPut_test3", type=str)
args = parser.parse_args()

if args.train == "train":
    train(args)
else:
    test(args)