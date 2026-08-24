"""
llm_demo_003.py

Study explicit temporal memory in SparsePBitLMTemporalMemory.

Question:
Does a longer reservoir-state history improve language quality while
keeping the 128-pbit reservoir fixed and without backpropagation through it?

Best configuration:
  history_size=8
  memory_scale=0.35
  use_initial_state=False

Requires: pip install "transformers[torch]"

Results
     Model  Trainable     Acc      PPL     Time
     p-kit      22176   0.793   12.155    1.67s
       GPT     103488   0.829    2.294    2.21s

The p-kit model reaches 79.3% accuracy vs 82.9% for the GPT baseline.
It uses 22,176 vs 103,488 trainable parameters, about 4.7x fewer.
There is no backpropagation through the p-bit reservoir.
Training time is similar in this small experiment.
"""

import math, time
import numpy as np
import torch
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

VOCAB=sorted(set(TRAIN)); C2I={c:i for i,c in enumerate(VOCAB)}
def encode(text): return [C2I[c] for c in text]

def evaluate_pkit(model):
    model.reset(); correct=total=0; nll=0
    for current,target in zip(TEST[:-1],TEST[1:]):
        p=model.next_char_probabilities(current,temperature=1.0)
        tid=model.char_to_id[target]
        correct+=int(np.argmax(p)==tid)
        nll-=math.log(max(float(p[tid]),1e-12)); total+=1
    return correct/total,math.exp(nll/total)

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
            correct+=int(torch.argmax(p)==ids[t])
            nll-=math.log(max(float(p[ids[t]]),1e-12)); total+=1
    return correct/total,math.exp(nll/total)

def main():
    np.random.seed(7); torch.manual_seed(7)

    print("Training p-kit model...")
    pkit=SparsePBitLMTemporalMemory(
        n_pbits=128,degree=8,input_fanout=16,
        recurrent_scale=.40,input_scale=1.4,
        memory_scale=.35,history_size=8,
        use_initial_state=False,seed=7)

    start=time.perf_counter()
    pkit.fit(TRAIN,washout=30,ridge=.05)
    pkit_time=time.perf_counter()-start
    pkit_acc,pkit_ppl=evaluate_pkit(pkit)

    print("Training GPT baseline...")
    gpt,gpt_time=train_gpt()
    gpt_acc,gpt_ppl=evaluate_gpt(gpt)

    print("\nResults")
    print(f"{'Model':>10} {'Trainable':>10} {'Acc':>7} {'PPL':>8} {'Time':>8}")
    print(f"{'p-kit':>10} {pkit.readout.size:10d} {pkit_acc:7.3f} {pkit_ppl:8.3f} {pkit_time:7.2f}s")
    print(f"{'GPT':>10} {sum(p.numel() for p in gpt.parameters()):10d} "
          f"{gpt_acc:7.3f} {gpt_ppl:8.3f} {gpt_time:7.2f}s")

if __name__=="__main__":
    main()