"""
llm_demo_003.py

Study memory depth in SparsePBitLMAdvanced. This is a version with explicit temporal memory.

Question:
Does a longer reservoir-state history improve language quality while
keeping the 128-pbit reservoir fixed and without backpropagation through it?

Requires: pip install "transformers[torch]"

Results
 History  Trainable     Acc      PPL     Time
       1       3360   0.561   13.730    1.62s
       2       6048   0.671   13.107    1.57s
       4      11424   0.768   12.555    2.02s
       8      22176   0.793   12.155    2.70s *
      16      43680   0.768   12.169    3.02s
     GPT     103488   0.829    2.294    2.17s #
     
The optimum here is history = 8.
We have 79.3% accuracy vs the base line GPT 82.9%.
22,176 vs 103,488 trainable parameters => 4.7× fewer then GPT.
No backpropagation through the p-bit reservoir.
Similar training time: 2.70s vs 2.17s.
"""

import math, time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from llm_model import SparsePBitLMTemporalMemory

TRAIN = """
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
""" * 20

TEST = """
the cat moved to the door.
the dog moved to the mat.
the robot looked at the cat.
"""

VOCAB = sorted(set(TRAIN))
C2I = {c:i for i,c in enumerate(VOCAB)}
def encode(text): return [C2I[c] for c in text]

def evaluate_pkit(model):
    model.reset(); correct=total=0; nll=0
    for current,target in zip(TEST[:-1],TEST[1:]):
        p=model.next_char_probabilities(current,temperature=1.0)
        tid=model.char_to_id[target]
        correct += int(np.argmax(p)==tid)
        nll -= math.log(max(float(p[tid]),1e-12)); total += 1
    return correct/total, math.exp(nll/total)

def train_gpt(context=32,epochs=15):
    ids=encode(TRAIN)
    blocks=[torch.tensor(ids[i:i+context]) for i in range(0,len(ids)-context,context//2)]
    model=GPT2LMHeadModel(GPT2Config(
        vocab_size=len(VOCAB),n_positions=context,n_ctx=context,
        n_embd=64,n_layer=2,n_head=2,
        resid_pdrop=0,embd_pdrop=0,attn_pdrop=0,
        bos_token_id=None,eos_token_id=None))
    loader=DataLoader(blocks,batch_size=16,shuffle=True)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-3)
    start=time.perf_counter(); model.train()
    for _ in range(epochs):
        for batch in loader:
            opt.zero_grad()
            loss=model(input_ids=batch,labels=batch).loss
            loss.backward(); opt.step()
    return model,time.perf_counter()-start

def evaluate_gpt(model,context=32):
    ids=encode(TEST); correct=total=0; nll=0; model.eval()
    with torch.no_grad():
        for t in range(1,len(ids)):
            x=torch.tensor(ids[max(0,t-context):t]).unsqueeze(0)
            p=torch.softmax(model(x).logits[0,-1],dim=-1)
            correct += int(torch.argmax(p)==ids[t])
            nll -= math.log(max(float(p[ids[t]]),1e-12)); total += 1
    return correct/total,math.exp(nll/total)

def main():
    np.random.seed(7); torch.manual_seed(7)

    print("Training GPT baseline...")
    gpt,gpt_time=train_gpt()
    gpt_acc,gpt_ppl=evaluate_gpt(gpt)
    gpt_params=sum(p.numel() for p in gpt.parameters())

    results=[]
    for h in (1,2,4,8,16):
        print(f"Training p-kit history={h}...")
        model=SparsePBitLMTemporalMemory(
            n_pbits=128,degree=8,input_fanout=16,
            recurrent_scale=.40,input_scale=1.4,
            memory_scale=.35,history_size=h,seed=7)

        start=time.perf_counter()
        model.fit(TRAIN,washout=30,ridge=.05)
        elapsed=time.perf_counter()-start
        acc,ppl=evaluate_pkit(model)
        results.append((h,model.readout.size,acc,ppl,elapsed))

    print("\nResults")
    print(f"{'History':>8} {'Trainable':>10} {'Acc':>7} {'PPL':>8} {'Time':>8}")
    for h,p,acc,ppl,t in results:
        print(f"{h:8d} {p:10d} {acc:7.3f} {ppl:8.3f} {t:7.2f}s")
    print(f"{'GPT':>8} {gpt_params:10d} {gpt_acc:7.3f} {gpt_ppl:8.3f} {gpt_time:7.2f}s")

    h=[r[0] for r in results]
    acc=[r[2] for r in results]
    plt.plot(h,acc,"o-",label="p-kit")
    plt.axhline(gpt_acc,linestyle="--",label="Tiny GPT")
    plt.xlabel("History depth"); plt.ylabel("Test accuracy")
    plt.title("p-kit memory-depth scaling")
    plt.xticks(h); plt.grid(); plt.legend(); plt.show()

if __name__ == "__main__":
    main()