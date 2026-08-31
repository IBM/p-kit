"""
llm_demo_005.py

p-kit LM vs tiny GPT using full tiny shakespeare:

A fixed stochastic p-bit recurrent architecture with substantially fewer trainable parameters
approaches a small GPT baseline on a real character-language task, while requiring gradient
training only for the readout.

NOTES:
   - It uses the full Tiny Shakespeare dataset
   - The comparison is fair: p-kit and GPT use the same train/validation/test split and vocabulary.
   - p-kit keeps a clear efficiency advantage: about 34% fewer trainable parameters.
   - The p-bit reservoir itself is fixed; no backpropagation goes through the recurrent p-kit network.
   - Only the final linear readout is trained with gradients, which is much simpler than end-to-end GPT training.
   - Performance is still reasonably close
   - It reports training time and inference
   - Validation is used only for temperature calibration, while the final numbers come from an untouched test set.
   
Most important LM p-kit parameters:

   - n_pbits — reservoir size
   - history_size — number of past reservoir states used by the readout
   - memory_scale — strength of recurrent state feedback
   - degree — reservoir connectivity
   - recurrent_scale — strength of recurrent couplings
   - input_fanout — how many p-bits each input character drives
   - input_scale — strength of the character input
   - ridge — ridge regularization of the initial readout
   - LAMBDA — distillation strength for the final readout
 
Results:
     Model  Trainable     T     Acc      PPL     Train     Infer
     p-kit      71500  0.85   0.299   13.642    78.28s     4.09s
       GPT     108352  1.50   0.355    9.557    38.22s     9.85s
       
     lm-p-kit uses 34.0% fewer trainable parameters.
     lm-p-kit inference is faster than GPT and could be optimized further.
     Memory usage can be optimized for lm-p-kit.
     
GPT achieves better accuracy and perplexity, while p-kit retains parameter-count and inference-speed advantages.
Faster language model p-kit inference suggests potential for lower energy consumption and better datacenter efficiency.
"""

import math,time
import numpy as np
import torch
import torch.nn.functional as F
from urllib.request import urlopen
from torch.utils.data import DataLoader
from transformers import GPT2Config,GPT2LMHeadModel
from p_kit.llm.llm_models import SparsePBitLMTemporalMemory

SEED=7
CONTEXT=64
LAMBDA=1.
TEMPS=[.1,.2,.3,.5,.7,.85,1.,1.2,1.5,2.]
TEACHER_TEMP=.1

def load_data():
    text=urlopen("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt").read().decode()
    n=int(len(text)*.9); train=text[:n]; rest=text[n:]; k=len(rest)//2
    valid,test=rest[:k],rest[k:]; vocab=sorted(set(train))
    valid="".join(c for c in valid if c in vocab); test="".join(c for c in test if c in vocab)
    return train,valid,test,vocab

def collect(model,text,washout=0):
    ids=[model.char_to_id[c] for c in text]; model.reset(); X=[]; y=[]
    for i in range(len(ids)-1):
        f=model._features(ids[i])
        if i>=washout: X.append(f); y.append(ids[i+1])
    return np.asarray(X,np.float32),np.asarray(y,np.int64)

def softmax(z,T=1.):
    z=z/T; z-=z.max(1,keepdims=True); p=np.exp(z)
    return p/p.sum(1,keepdims=True)

def evaluate(z,y,T=1.):
    p=softmax(z,T)
    acc=np.mean(np.argmax(p,1)==y)
    ppl=math.exp(-np.mean(np.log(np.maximum(p[np.arange(len(y)),y],1e-12))))
    return acc,ppl

def calibrate(z,y):
    return min(TEMPS,key=lambda T:evaluate(z,y,T)[1])

def train_ridge(X,y,n_classes,ridge=.05):
    Y=np.eye(n_classes,dtype=np.float32)[y]
    A=X.T@X+ridge*np.eye(X.shape[1],dtype=np.float32)
    return np.linalg.solve(A,X.T@Y)

def train_distilled(X,y,W0,lam=LAMBDA,epochs=2):
    x=torch.from_numpy(X); t=torch.from_numpy(y)
    teacher=torch.from_numpy(softmax(X@W0,TEACHER_TEMP)).float()
    layer=torch.nn.Linear(X.shape[1],W0.shape[1],bias=False)
    with torch.no_grad(): layer.weight.copy_(torch.from_numpy(W0.T).float())
    opt=torch.optim.AdamW(layer.parameters(),lr=1e-3,weight_decay=1e-4)

    for _ in range(epochs):
        for idx in torch.randperm(len(t)).split(512):
            z=layer(x[idx])
            ce=F.cross_entropy(z,t[idx])
            kd=-(teacher[idx]*F.log_softmax(z,1)).sum(1).mean()
            opt.zero_grad(); (ce+lam*kd).backward(); opt.step()

    return layer.weight.detach().numpy().T

def train_gpt(train,vocab,epochs=5):
    c2i={c:i for i,c in enumerate(vocab)}
    ids=[c2i[c] for c in train]
    blocks=[torch.tensor(ids[i:i+CONTEXT]) for i in range(0,len(ids)-CONTEXT,CONTEXT)]

    model=GPT2LMHeadModel(GPT2Config(
        vocab_size=len(vocab),n_positions=CONTEXT,n_ctx=CONTEXT,
        n_embd=64,n_layer=2,n_head=2,
        resid_pdrop=0,embd_pdrop=0,attn_pdrop=0,
        bos_token_id=None,eos_token_id=None))

    loader=DataLoader(blocks,batch_size=32,shuffle=True)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-3)

    for _ in range(epochs):
        for x in loader:
            loss=model(input_ids=x,labels=x).loss
            opt.zero_grad(); loss.backward(); opt.step()

    return model,c2i

def gpt_logits(model,text,c2i,batch=256):
    ids=torch.tensor([c2i[c] for c in text])
    X=ids[:-1].unfold(0,CONTEXT,1); y=ids[CONTEXT:]
    out=[]; model.eval()

    with torch.no_grad():
        for i in range(0,len(X),batch):
            out.append(model(X[i:i+batch]).logits[:,-1].cpu())

    return torch.cat(out).numpy(),y.numpy()

def main():
    np.random.seed(SEED); torch.manual_seed(SEED)
    train,valid,test,vocab=load_data()
    print(f"Train {len(train):,}, valid {len(valid):,}, test {len(test):,}, vocab {len(vocab)}")

    model=SparsePBitLMTemporalMemory(
        n_pbits=128,degree=8,input_fanout=16,
        recurrent_scale=.40,input_scale=1.4,
        memory_scale=.35,history_size=8,
        use_initial_state=False,seed=SEED)

    model._build_vocab(train)

    t=time.perf_counter()
    X,y=collect(model,train,100)
    ridge=train_ridge(X,y,len(vocab),.05)
    model.readout=ridge
    W=train_distilled(X,y,ridge)
    ptrain=time.perf_counter()-t

    Xv,yv=collect(model,valid,CONTEXT-1)
    T=calibrate(Xv@W,yv)

    t=time.perf_counter()
    Xt,yt=collect(model,test,CONTEXT-1)
    pacc,pppl=evaluate(Xt@W,yt,T)
    pinfer=time.perf_counter()-t

    t=time.perf_counter()
    gpt,c2i=train_gpt(train,vocab)
    gtrain=time.perf_counter()-t

    gv,gy=gpt_logits(gpt,valid,c2i)
    gT=calibrate(gv,gy)

    t=time.perf_counter()
    gt,gty=gpt_logits(gpt,test,c2i)
    gacc,gppl=evaluate(gt,gty,gT)
    ginfer=time.perf_counter()-t

    pp=W.size
    gp=sum(p.numel() for p in gpt.parameters())

    print(f"\n{'Model':>10} {'Trainable':>10} {'T':>5} {'Acc':>7} {'PPL':>8} {'Train':>9} {'Infer':>9}")
    print(f"{'p-kit':>10} {pp:10d} {T:5.2f} {pacc:7.3f} {pppl:8.3f} {ptrain:8.2f}s {pinfer:8.2f}s")
    print(f"{'GPT':>10} {gp:10d} {gT:5.2f} {gacc:7.3f} {gppl:8.3f} {gtrain:8.2f}s {ginfer:8.2f}s")
    print(f"\np-kit uses {100*(1-pp/gp):.1f}% fewer trainable parameters.")

if __name__=="__main__":
    main()