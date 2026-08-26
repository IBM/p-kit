"""
llm_demo_004.py

Compares SparsePBitLMTemporalMemory using CaSuDaSolver on:
  1. NumpyBackend (CPU)
  2. TorchBackend (CPU, and CUDA if available)
  3. CupyBackend (CUDA, if available)
against a Tiny GPT baseline, on both CPU and GPU (if available).

The CSD algorithm is a single implementation (CaSuDaSolver); only the
backend changes between rows below.

Best p-kit configuration:
  history_size=8
  memory_scale=0.35
  use_initial_state=False

Results
                Model  Trainable     Acc      PPL     Time
     CaSuDaSolver CPU      22176   0.793   12.155    1.97s
TorchCaSuDaSolver CPU      22176   0.756   12.125    9.24s
              GPT CPU     103488   0.829    2.294    2.07s

    The Torch backend is currently slower on CPU than the NumPy backend,
    but it might get faster once we switch to a bigger p-kit LM and CUDA.

"""

import math,time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config,GPT2LMHeadModel

from llm_model import SparsePBitLMTemporalMemory
from p_kit.solver.csd_solver import CaSuDaSolver
from p_kit.backends import NumpyBackend, CupyBackend, TorchBackend

TRAIN="""
the cat sat on the mat.
the dog sat by the door.
the cat looked at the dog.
the dog looked at the cat.
the cat walked to the door.
the dog walked to the mat.
the small robot saw the cat.
the small robot saw the dog.
the robot moved to the door.
the robot moved to the mat.
"""*20

TEST="""
the cat moved to the door.
the dog moved to the mat.
the robot looked at the cat.
"""

VOCAB=sorted(set(TRAIN)); C2I={c:i for i,c in enumerate(VOCAB)}
def encode(text): return [C2I[c] for c in text]

try:
    import cupy as cp
    CUPY_CUDA=cp.cuda.runtime.getDeviceCount()>0
except Exception:
    cp=None
    CUPY_CUDA=False

TORCH_CUDA=torch.cuda.is_available()

def evaluate_pkit(model):
    model.reset(); correct=total=0; nll=0
    for current,target in zip(TEST[:-1],TEST[1:]):
        p=model.next_char_probabilities(current,temperature=1.0)
        tid=model.char_to_id[target]
        correct+=int(np.argmax(p)==tid)
        nll-=math.log(max(float(p[tid]),1e-12)); total+=1
    return correct/total,math.exp(nll/total)

def make_pkit():
    return SparsePBitLMTemporalMemory(
        n_pbits=128,degree=8,input_fanout=16,
        recurrent_scale=.40,input_scale=1.4,
        memory_scale=.35,history_size=8,
        use_initial_state=False,seed=7)

def train_pkit(model,sync=None):
    if sync: sync()
    start=time.perf_counter()
    model.fit(TRAIN,washout=30,ridge=.05)
    if sync: sync()
    elapsed=time.perf_counter()-start
    acc,ppl=evaluate_pkit(model)
    return acc,ppl,elapsed

def make_gpt(context=32,device="cpu"):
    model=GPT2LMHeadModel(GPT2Config(
        vocab_size=len(VOCAB),n_positions=context,n_ctx=context,
        n_embd=64,n_layer=2,n_head=2,
        resid_pdrop=0,embd_pdrop=0,attn_pdrop=0,
        bos_token_id=None,eos_token_id=None))
    return model.to(device)

def train_gpt(device="cpu",context=32,epochs=15):
    ids=encode(TRAIN)
    blocks=[torch.tensor(ids[i:i+context]) for i in range(0,len(ids)-context,context//2)]
    model=make_gpt(context,device)
    loader=DataLoader(blocks,batch_size=16,shuffle=True)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-3)

    if device=="cuda": torch.cuda.synchronize()
    start=time.perf_counter(); model.train()

    for _ in range(epochs):
        for batch in loader:
            batch=batch.to(device)
            opt.zero_grad()
            loss=model(input_ids=batch,labels=batch).loss
            loss.backward(); opt.step()

    if device=="cuda": torch.cuda.synchronize()
    return model,time.perf_counter()-start

def evaluate_gpt(model,device="cpu",context=32):
    ids=encode(TEST); correct=total=0; nll=0; model.eval()
    with torch.no_grad():
        for t in range(1,len(ids)):
            x=torch.tensor(ids[max(0,t-context):t],device=device).unsqueeze(0)
            p=torch.softmax(model(x).logits[0,-1],dim=-1)
            correct+=int(torch.argmax(p).item()==ids[t])
            nll-=math.log(max(float(p[ids[t]].item()),1e-12)); total+=1
    return correct/total,math.exp(nll/total)

def add_pkit(results,name,backend,sync=None,cache_J=False):
    print(f"Training {name}...")
    model=make_pkit()
    model.reservoir_solver=CaSuDaSolver(
        Nt=10,dt=.1667,i0=.8,seed=7,backend=backend,cache_J=cache_J)
    model.relu_solver=CaSuDaSolver(
        Nt=10,dt=.1667,i0=.8,seed=8,backend=backend,cache_J=cache_J)
    acc,ppl,t=train_pkit(model,sync)
    results.append((name,model.readout.size,acc,ppl,t))

def add_gpt(results,name,device):
    print(f"Training {name}...")
    torch.manual_seed(7)
    if device=="cuda": torch.cuda.manual_seed_all(7)
    model,t=train_gpt(device)
    acc,ppl=evaluate_gpt(model,device)
    results.append((name,sum(p.numel() for p in model.parameters()),acc,ppl,t))

def main():
    np.random.seed(7); torch.manual_seed(7)
    results=[]
    
    add_pkit(results, "CaSuDaSolver CPU", NumpyBackend())
    add_pkit(results, "TorchCaSuDaSolver CPU", TorchBackend(device="cpu"), cache_J=True)
    add_gpt(results, "GPT CPU", "cpu")

    if CUPY_CUDA:
        add_pkit(
            results, "CaSuDaSolver CUDA", CupyBackend(),
            sync=lambda: cp.cuda.Stream.null.synchronize())

    if TORCH_CUDA:
        add_pkit(
            results, "TorchCaSuDaSolver CUDA", TorchBackend(device="cuda", compile=True),
            sync=torch.cuda.synchronize, cache_J=True)
        add_gpt(results, "GPT CUDA", "cuda")

    print("\nCUDA availability")
    print(f"CuPy:   {CUPY_CUDA}")
    print(f"PyTorch:{TORCH_CUDA}")

    print("\nResults")
    print(f"{'Model':>20} {'Trainable':>10} {'Acc':>7} {'PPL':>8} {'Time':>8}")
    for name,p,acc,ppl,t in results:
        print(f"{name:>20} {p:10d} {acc:7.3f} {ppl:8.3f} {t:7.2f}s")

if __name__=="__main__":
    main()