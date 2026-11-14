import os,sys
os.chdir('/content/drive/MyDrive/DSGA1011_A4/part2_run_20261112_ft')
sys.path.insert(0,os.getcwd())
from utils import save_queries_and_records,compute_metrics
import torch
DEVICE=torch.device('cuda')
import torch,gc,json,random,numpy as np
from pathlib import Path
from transformers import T5TokenizerFast,T5Config,T5ForConditionalGeneration,get_cosine_schedule_with_warmup
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
for n in ['fixed_model','model_w','model_mix','model_refit']:
 if n in globals():del globals()[n]
gc.collect();torch.cuda.empty_cache();torch.manual_seed(42);random.seed(42);np.random.seed(42)
nt=T5TokenizerFast.from_pretrained('google-t5/t5-small')
class NativeDS(Dataset):
 def __init__(self,split):
  self.x=nt([s.strip() for s in Path(f'data/{split}.nl').read_text().splitlines()],add_special_tokens=True)['input_ids']
  self.y=None if split=='test' else nt([s.strip().replace('<','lessthan') for s in Path(f'data/{split}.sql').read_text().splitlines()],add_special_tokens=True)['input_ids']
 def __len__(self):return len(self.x)
 def __getitem__(self,i):return self.x[i],None if self.y is None else self.y[i]
def nc(rows):
 x=pad_sequence([torch.tensor(r[0]) for r in rows],batch_first=True,padding_value=0)
 y=None if rows[0][1] is None else pad_sequence([torch.tensor(r[1]) for r in rows],batch_first=True,padding_value=-100)
 return x,x.ne(0).long(),y
ndl={s:DataLoader(NativeDS(s),batch_size=16,shuffle=s=='train',collate_fn=nc) for s in ['train','dev','test']}
nm=T5ForConditionalGeneration(T5Config.from_pretrained('google-t5/t5-small')).to(DEVICE)
no=torch.optim.AdamW(nm.parameters(),lr=1e-4,weight_decay=0)
ns=get_cosine_schedule_with_warmup(no,2*len(ndl['train']),100*len(ndl['train']))
nd=Path('checkpoints/scr_experiments/native_lt_v1');nd.mkdir(parents=True,exist_ok=True);nt.save_pretrained(nd/'tokenizer')
@torch.no_grad()
def ng(loader):
 nm.eval();qs=[]
 for x,m,_ in loader:
  ids=nm.generate(input_ids=x.to(DEVICE),attention_mask=m.to(DEVICE),num_beams=4,max_new_tokens=512)
  qs.extend(s.replace('lessthan','<').replace('\n',' ').strip() for s in nt.batch_decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False))
 return qs
best=-1;hist=[]
print('NATIVE_RANDOM_INIT_START',len(ndl['train'].dataset),flush=True)
for ep in range(100):
 nm.train();tot=cnt=0
 for x,m,y in ndl['train']:
  x=x.to(DEVICE);m=m.to(DEVICE);y=y.to(DEVICE);no.zero_grad(set_to_none=True)
  with torch.autocast('cuda',dtype=torch.bfloat16):loss=nm(input_ids=x,attention_mask=m,labels=y).loss
  loss.backward();torch.nn.utils.clip_grad_norm_(nm.parameters(),1.0,error_if_nonfinite=True);no.step();ns.step()
  k=y.ne(-100).sum().item();tot+=loss.item()*k;cnt+=k
 print('NATIVE_EPOCH',ep+1,'LOSS',tot/cnt,flush=True)
 if (ep+1)%5==0:
  qs=ng(ndl['dev']);sp='results/native_lt_dev.sql';rp='records/native_lt_dev.pkl';save_queries_and_records(qs,sp,rp)
  mm=compute_metrics('data/dev.sql',sp,'records/ground_truth_dev.pkl',rp);f1=float(mm[2]);hist.append({'epoch':ep+1,'f1':f1,'loss':tot/cnt});(nd/'history.json').write_text(json.dumps(hist,indent=2))
  print('NATIVE_DEV',ep+1,mm[:3],flush=True)
  if f1>best:best=f1;nm.save_pretrained(nd/'best');print('NATIVE_BEST',best,flush=True)
 nm.save_pretrained(nd/'last')
print('NATIVE_TRAIN_DONE',best,flush=True)