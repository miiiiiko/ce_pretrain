import os, sys, time, json, ljqpy, random
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

# userdir = '/data1/ljq'
# sys.path.append(userdir)
import pt_utils
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
seed_everything(42)

configs = {'tnews': {'xkey':'sentence', 'ykey':'label_desc'},
            'iflytek': {'xkey':'sentence', 'ykey':'label_des'},
            'wsc': {'xkey':'text', 'ykey':'label', 'epochs':20},
            'afqmc': {'xkey':('sentence1', 'sentence2'), 'ykey':'label'},
            'cmnli': {'xkey':('sentence1', 'sentence2'), 'ykey':'label'},
            'csl': {'xkey':('keyword', 'abst'), 'ykey':'label', 'maxlen': 256, 'epochs':5}
            }
baseline = {'afqmc':74.04,'tnews':56.94,'iflytek':60.31,'cmnli':80.51,'wsc':67.8, 'csl':81.0}

# https://github.com/CLUEbenchmark/CLUE

def tostr(x):
    if type(x) is list: return ' '.join(x)
    return x 

def wsc_processor(line):
    text_a = line['text']
    text_a_list = list(text_a)
    target = line['target']
    query = target['span1_text']
    query_idx = target['span1_index']
    pronoun = target['span2_text']
    pronoun_idx = target['span2_index']
    assert text_a[pronoun_idx: (pronoun_idx + len(pronoun))] == pronoun, "pronoun: {}".format(pronoun)
    assert text_a[query_idx: (query_idx + len(query))] == query, "query: {}".format(query)
    if pronoun_idx > query_idx:
        text_a_list.insert(query_idx, "_")
        text_a_list.insert(query_idx + len(query) + 1, "_")
        text_a_list.insert(pronoun_idx + 2, "[")
        text_a_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
    else:
        text_a_list.insert(pronoun_idx, "[")
        text_a_list.insert(pronoun_idx + len(pronoun) + 1, "]")
        text_a_list.insert(query_idx + 2, "_")
        text_a_list.insert(query_idx + len(query) + 2 + 1, "_")
    text_a = "".join(text_a_list)
    return text_a

class ClassifyDataset(torch.utils.data.Dataset):
    def __init__(self, data, config, label2id) -> None:
        self.data = data
        xkey, ykey = config['xkey'], config['ykey']
        self.to_tuple = lambda z:(z[xkey], label2id[z[ykey]])
        if type(xkey) is tuple:
            self.to_tuple = lambda z:(tuple(tostr(z[xk]) for xk in xkey), label2id[z[ykey]])
        if 'wsc' in sys.argv:
            self.to_tuple = lambda z:(wsc_processor(z), label2id[z[ykey]])
    def __len__(self): return len(self.data)
    def __getitem__(self, k):
        item = self.data[k]
        return self.to_tuple(item)

def load_dataset(name):
    dr = f'/mnt/data122/qsm/CLUEdatasets/{name}'
    config = configs[name]
    ykey = config['ykey']
    trains = ljqpy.LoadJsons(os.path.join(dr, 'train.json'))
    valids = ljqpy.LoadJsons(os.path.join(dr, 'dev.json'))
    labelds = defaultdict(int)
    for sample in valids: labelds[sample[ykey]] += 1
    labels = ljqpy.FreqDict2List(labelds)
    id2label = [x[0] for x in labels]
    label2id = {v:k for k,v in enumerate(id2label)}
    return ClassifyDataset(trains, config, label2id), \
            ClassifyDataset(valids, config, label2id), id2label

def collate_fn(xs):
    xx = tokenizer([x[0] for x in xs], return_tensors='pt', truncation=True, padding=True, max_length=maxlen)['input_ids']
    yy = torch.LongTensor([x[1] for x in xs])
    return xx, yy

def collate_fn_pair(xs):
    xx = tokenizer([x[0][0] for x in xs], [x[0][1] for x in xs], return_tensors='pt', truncation='only_second', padding=True, max_length=maxlen)
    yy = torch.LongTensor([x[1] for x in xs])
    return xx['input_ids'], xx['token_type_ids'], yy

class Classifier(nn.Module):
	def __init__(self, encoder, n_tags, cls_only=True) -> None:
		super().__init__()
		self.n_tags = n_tags
		self.encoder = encoder
		self.fc = nn.Linear(768, n_tags)
		self.cls_only = cls_only    
	def forward(self, x, seg=None):
		if seg is None: seg = torch.zeros_like(x)
		z = self.encoder(x, token_type_ids=seg).last_hidden_state
		if self.cls_only: z = z[:,0]
		out = self.fc(z)
		return out

loss_func = nn.CrossEntropyLoss()
def train_func(model, ditem):
    ditem = [x.cuda() for x in ditem]
    yy = ditem[-1]
    seg = ditem[1] if len(ditem) == 3 else None
    zz = model(ditem[0], seg=seg)
    loss = loss_func(zz, yy)
    return {'loss': loss}

def test_func(): 
    global accu
    yt, yp = [], []
    model.eval()
    with torch.no_grad():
        for ditem in dl_valid:
            seg = ditem[1].cuda() if len(ditem) == 3 else None
            xx, yy = ditem[0].cuda(), ditem[-1]
            zz = model(xx, seg=seg)
            zz = zz.detach().cpu().argmax(-1)
            for y in yy: yt.append(y.item())
            for z in zz: yp.append(z.item())
    accu = (np.array(yt) == np.array(yp)).sum() / len(yp)
    print(f'Accu: {accu:.4f}')
    model.train()
    return accu

name = sys.argv[1]

if type(configs[name]['xkey']) is tuple: collate_fn = collate_fn_pair
maxlen = configs[name].get('maxlen', 128)

ds_train, ds_valid, id2label = load_dataset(name)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=32, collate_fn=collate_fn, shuffle=True)
dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=32, collate_fn=collate_fn)

sys.path.append('../plm_trainer/')

#################################################
from cetokenizer import CEBertTokenizer, CEBertTokenizerFast
from transformers import BertTokenizer, BertModel, BertConfig
plm = 'hfl/chinese-roberta-wwm-ext'

tokenizer = CEBertTokenizerFast('plm_trainer/vocab.txt')
config = BertConfig.from_pretrained(plm)
config.vocab_size = len(tokenizer.vocab)
model = Classifier(BertModel(config), len(id2label))
model.encoder.load_state_dict(torch.load('plm_trainer/myroberta_5.pt'), strict=False)
#################################################

#tokenizer = BertTokenizer.from_pretrained(plm)
#model = Classifier(BertModel.from_pretrained(plm), len(id2label))

def train():
    model.cuda()
    epochs = configs[name].get('epochs', 3)
    total_steps = len(dl_train) * epochs
    optimizer, scheduler = pt_utils.get_bert_optim_and_sche(model, 2e-5, total_steps)
    pt_utils.train_model(model, optimizer, dl_train, epochs, train_func, test_func, scheduler=scheduler)
    print(f'\nAccuracy: {accu*100:.2f}, Baseline {name}: {baseline[name]:.2f}')

train()
