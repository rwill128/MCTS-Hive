#!/usr/bin/env python3
"""Hive‑Zero: continuous self‑play + online training.

Run (CPU):
    python hive_zero.py
Run (GPU):
    python hive_zero.py --gpu --games 50 --epochs 2000
"""
from __future__ import annotations

import argparse, random, pathlib, time
from collections import deque
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from HivePocket.HivePocket import HiveGame

# ---------------- Board & encoding --------------------------
BOARD_R = 6                   # radius → 13 × 13 square
H = W = BOARD_R * 2 + 1
PIECE_TYPES = ["Q", "B", "S", "A", "G"]
C = len(PIECE_TYPES) * 2 + 1  # plus side‑to‑move
shift = BOARD_R

def to_xy(q: int, r: int):
    return r + shift, q + shift

def encode_state(state: dict, perspective: str) -> torch.Tensor:
    t = torch.zeros((C, H, W))
    side = 0 if state["current_player"] == perspective else 1
    for (q, r), stack in state["board"].items():
        if not stack: continue
        owner, insect = stack[-1]
        if insect[0] not in PIECE_TYPES: continue
        idx = PIECE_TYPES.index(insect[0]) + (0 if owner == perspective else len(PIECE_TYPES))
        y, x = to_xy(q, r)
        if 0 <= y < H and 0 <= x < W:
            t[idx, y, x] = 1.0
    t[-1].fill_(side)
    return t

# ---------------- Network -----------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(c)
    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)

class HiveZeroNet(nn.Module):
    def __init__(self, ch=64, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(C, ch, 3, 1, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU())
        self.res = nn.Sequential(*[ResidualBlock(ch) for _ in range(blocks)])
        self.p_head = nn.Sequential(nn.Conv2d(ch, 2, 1), nn.BatchNorm2d(2), nn.ReLU(),
                                    nn.Flatten(), nn.Linear(2*H*W, H*W*len(PIECE_TYPES)))
        self.v_head = nn.Sequential(nn.Conv2d(ch, 1, 1), nn.BatchNorm2d(1), nn.ReLU(),
                                    nn.Flatten(), nn.Linear(H*W,64), nn.ReLU(), nn.Linear(64,1), nn.Tanh())
    def forward(self,x):
        x=self.res(self.stem(x))
        return self.p_head(x), self.v_head(x).squeeze(1)

# ---------------- Evaluator ---------------------------------
class ZeroEvaluator:
    """Wraps the policy‑value net for quick (logits, value) inference."""
    def __init__(self, model: HiveZeroNet, device: str = "cpu"):
        self.model = model.to(device)
        self.dev = device
    @torch.no_grad()
    def evaluate(self, state: dict, perspective: str):
        x = encode_state(state, perspective).unsqueeze(0).to(self.dev)
        logits, val = self.model(x)
        return logits.cpu().numpy()[0], val.item()

# ---------------- Action indexing ---------------------------
AXIAL_TO_INDEX, FLAT = {}, []
for q in range(-BOARD_R, BOARD_R+1):
    for r in range(-BOARD_R, BOARD_R+1):
        for t in PIECE_TYPES:
            AXIAL_TO_INDEX[(q,r,t)] = len(FLAT)
            FLAT.append((q,r,t))
POLICY_SIZE = len(FLAT)  # 845

# ---------------- Self‑play helpers -------------------------

def softmax_T(x, T):
    z=np.exp((x-x.max())/T); return z/z.sum()

def mask_illegal(priors, legal):
    """Zero illegal indexes, renormalise; never return NaNs."""
    mask = np.zeros_like(priors)
    legal_idxs = []
    for a in legal:
        if a[0] == "PASS":
            continue  # no PASS plane yet
        try:
            if a[0] == "PLACE":
                _, tp, (q, r) = a
                idx = AXIAL_TO_INDEX[(q, r, tp[0])]
            else:  # MOVE
                q, r = a[2]
                idx = AXIAL_TO_INDEX.get((q, r, PIECE_TYPES[0]))
            if idx is not None:
                mask[idx] = 1.0
                legal_idxs.append(idx)
        except KeyError:
            continue

    if mask.sum() == 0:
        # All legal moves outside bounding box – fall back to uniform on random subset
        if not legal_idxs:
            priors = np.full_like(priors, 1 / priors.size)
            return priors
        mask[legal_idxs] = 1.0

    priors = priors * mask
    total = priors.sum()
    if total == 0 or np.isnan(total):
        priors = mask / mask.sum()
    else:
        priors /= total
    return priors

def flat_to_action(idx, legal):
    q,r,t=FLAT[idx]
    for a in legal:
        if a[0]=="PLACE": _,tp,(q0,r0)=a;
        elif a[0]=="MOVE": dest=a[2]; q0,r0=dest
        else: continue
        if (q0,r0)==(q,r) and (a[0]=="MOVE" or tp[0]==t): return a
    return random.choice(legal)

# ---------------- Self‑play game ----------------------------

def play_one_game(model,T=1.0,max_moves=300):
    game=HiveGame(); state=game.getInitialState(); evalr=ZeroEvaluator(model)
    hist=[]
    while not game.isTerminal(state) and len(hist)<max_moves:
        logits,_=evalr.evaluate(state,state["current_player"])
        pri=mask_illegal(softmax_T(logits,T),game.getLegalActions(state))
        hist.append((game.copyState(state),pri,0))
        idx=np.random.choice(POLICY_SIZE,p=pri)
        action=flat_to_action(idx,game.getLegalActions(state))
        state=game.applyAction(state,action)
    winner=game.getGameOutcome(state); z=0 if winner=="Draw" else (1 if winner=="Player1" else -1)
    if state["current_player"]=="Player2": z=-z
    return [(s,p,z) for s,p,_ in hist]

# ---------------- Training ----------------------------------

def batch_tensors(batch,device):
    S=torch.stack([encode_state(s,"Player1") for s,_,_ in batch]).to(device)
    P=torch.tensor(np.array([p for _,p,_ in batch]),dtype=torch.float32,device=device)
    V=torch.tensor([v for _,_,v in batch],dtype=torch.float32,device=device)
    return S,P,V

def train_step(net,batch,opt,device):
    S,P,V=batch_tensors(batch,device)
    logits, vpred = net(S)
    loss=F.cross_entropy(logits,P.argmax(1))+F.mse_loss(vpred.squeeze(),V)
    opt.zero_grad(); loss.backward(); opt.step(); return loss.item()

# ---------------- Main loop ---------------------------------

def run(args):
    dev="cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    net=HiveZeroNet().to(dev); opt=optim.Adam(net.parameters(),lr=1e-3)
    buf:deque=deque(maxlen=args.buffer)
    print("Bootstrapping",args.games,"games …")
    for g in range(args.games):
        buf.extend(play_one_game(net,T=args.temp)); print(f"  game {g+1}/{args.games} → buffer {len(buf)}")
    ckdir=pathlib.Path(args.ckpt_dir); ckdir.mkdir(exist_ok=True)
    for ep in range(1,args.epochs+1):
        buf.extend(play_one_game(net,T=args.temp))
        if len(buf)>=args.batch:
            loss=train_step(net,random.sample(buf,args.batch),opt,dev)
            if ep%args.log_every==0: print(f"epoch {ep} | loss {loss:.4f} | buf {len(buf)}")
        if ep%args.ckpt_every==0:
            torch.save(net.state_dict(),ckdir/f"chkpt_{ep:05d}.pt")

# ---------------- Arg‑parser & entry ------------------------

def parser():
    p=argparse.ArgumentParser();
    p.add_argument("--gpu",action="store_true"); p.add_argument("--games",type=int,default=20)
    p.add_argument("--epochs",type=int,default=1000); p.add_argument("--buffer",type=int,default=50000)
    p.add_argument("--batch",type=int,default=256); p.add_argument("--temp",type=float,default=1.0)
    p.add_argument("--ckpt-dir",default="checkpoints"); p.add_argument("--ckpt-every",type=int,default=100)
    p.add_argument("--log-every",type=int,default=10); return p

if __name__=="__main__": run(parser().parse_args())
