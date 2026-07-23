#!/usr/bin/env python3
"""
DAGeR v13: DB-Optimized Iterative Feedback Refinement
===========================================================
v12 -> v13: DB-inspired runtime optimizations from the literature:

  1. EDDY GATE (Avnur SIGMOD 2000): delta-driven early termination
     If |S_struct - S_emb| < epsilon for all entities after round r, skip r+1.

  2. PROGRESSIVE REFINEMENT (Papadakis ICDE 2023 Sec IV-B):
     Round-aware delta threshold narrowing: eps1=0.15 -> eps2=0.05.
     Only entities with delta>eps_r are recomputed.

  3. RUNTIME DYNAMIC OPTIMIZATION (SIGMOD 2023):
     Cache anchor embeddings once per batch (was 3x redundant lookup).
     Pre-pad bridge_w once (was 3x padding).

Expected speedup: 35-45% per epoch (140s -> ~90s)
"""
import os, math, time, argparse, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dager_core import (IDFWeightedSignatures, ParametricSimulation, MultiConstraintBridge,
                       read_triples, build_ent_rel_mappings, compute_node_degrees, set_seed,
                       FusionDataset, collate_fn, NegativeSampler,
                       GMCBlock, DomainGate, BaseKGScorer, TransEScorer, RotatEScorer, ExeFuseScorer, SCORER_REGISTRY,
                       init_bert_scorer_weights)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

class StructuralEnhancement(nn.Module):
    """V13: DB-inspired optimization of iterative refinement.
    DB->KG mapping:
      - Eddy operator     -> delta-gated early termination
      - Progressive ER    -> epsilon-narrowing per round
      - Runtime re-opt    -> cached intermediate results (ae, padded_w)
    """
    def __init__(self, dim, max_k=8, dropout=0.1, lambda_bridge=0.2, n_rounds=3):
        super().__init__()
        self.dim = dim; self.max_k = max_k
        self.lambda_bridge = lambda_bridge; self.n_rounds = n_rounds
        self.gmc = GMCBlock(dim)
        self.gate = DomainGate(dim, dropout, use_relation=True)
        self.tau = nn.Parameter(torch.tensor(0.0))
        self.alpha_min = 0.05; self.alpha_max = 0.90
        self.alpha_raw = nn.Parameter(torch.tensor(0.5))
        self.alpha_deg = nn.Parameter(torch.tensor(-0.5))
        self.struct_attn = nn.Sequential(
            nn.Linear(max_k, max_k * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(max_k * 2, max_k), nn.Softmax(dim=-1))
        self.round_weights = nn.Parameter(torch.ones(n_rounds) / n_rounds)
        self.delta_mod = nn.Sequential(
            nn.Linear(1, dim // 4), nn.GELU(), nn.Linear(dim // 4, 1), nn.Sigmoid())
        self.round_alpha_bias = nn.Parameter(torch.zeros(n_rounds))
        self.register_buffer('delta_eps', torch.tensor([0.15, 0.05, 0.0]), persistent=False)
        self.eddy_eps = 0.03
        self.bridge = None

    def set_bridge(self, bridge): self.bridge = bridge
    def get_alpha(self):
        a = torch.sigmoid(self.alpha_raw).item()
        return self.alpha_min + (self.alpha_max - self.alpha_min) * a
    def get_alpha_for_degree(self, deg):
        with torch.no_grad():
            a = torch.sigmoid(self.alpha_raw + self.alpha_deg * math.log(1 + deg)).item()
            return self.alpha_min + (self.alpha_max - self.alpha_min) * a

    def compute_alpha(self, deg, delta=None, round_idx=0):
        deg_factor = torch.log(1.0 + deg.float())
        alpha_raw = torch.sigmoid(self.alpha_raw + self.alpha_deg * deg_factor)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha_raw
        if delta is not None:
            d_norm = delta.detach()
            dw = torch.sigmoid(d_norm * 3.0 - 1.5)
            bias = self.round_alpha_bias[round_idx].sigmoid()
            alpha = alpha + dw * (0.5 - alpha) * bias
            alpha = alpha.clamp(self.alpha_min, self.alpha_max)
        return alpha


    def _cache_anchors(self, bridge_w, bridge_idx, scorer):
        if bridge_idx is None:
            return None, None, None
        B, K_raw = bridge_w.shape[0], bridge_w.shape[1]
        if K_raw < self.max_k:
            pad = torch.zeros(B, self.max_k - K_raw, device=bridge_w.device)
            w = torch.cat([bridge_w, pad], dim=1)
        elif K_raw > self.max_k:
            w = bridge_w[:, :self.max_k]
        else:
            w = bridge_w
        ae = scorer.ent(bridge_idx)
        return w, ae, bridge_idx

    def _struct_score(self, w, delta_factor=None):
        if w is None:
            return torch.zeros(1, device=self.tau.device)
        w_cur = w
        if delta_factor is not None:
            w_cur = w_cur * (1.0 + delta_factor.unsqueeze(-1))
        attn = self.struct_attn(w_cur)
        return torch.sigmoid((attn * w_cur).sum(dim=-1))

    def _enrich(self, tg, re, cached_ae, cached_w, bridge_idx, scorer, delta=None):
        if bridge_idx is None or cached_ae is None:
            return tg
        ae = cached_ae
        gw = self.gate(tg, ae, r=re)
        if delta is not None:
            sharp = torch.sigmoid(delta.unsqueeze(-1) * 2.0)
            bvar = cached_w.var(dim=-1, keepdim=True)
            gw = gw * (1.0 + sharp * bvar / (bvar.mean() + 1e-8))
        borrowed = (gw.unsqueeze(-1) * ae).sum(dim=1)
        return tg + self.lambda_bridge * borrowed

    def forward(self, he, re, te, base_scorer,
                raw_head=None, raw_tail=None, raw_rel=None, tail_deg=None):
        B = he.shape[0]; device = he.device
        x = torch.stack([he, te], dim=1)
        adj = torch.ones(B, 2, 2, device=device)
        x = self.gmc(x, adj)
        hg, tg = x[:, 0, :], x[:, 1, :]

        bridge_w = bridge_idx = None
        if raw_tail is not None and self.bridge is not None:
            bridge_w, bridge_idx = self.bridge.lookup(raw_tail)
        cached_w, cached_ae, cached_bi = self._cache_anchors(bridge_w, bridge_idx, base_scorer)

        deg_tensor = tail_deg if tail_deg is not None else torch.ones(B, device=device) * 5.0
        base_alpha = self.compute_alpha(deg_tensor)

        # Round 0: baseline
        s0 = self._struct_score(cached_w)
        tg0 = self._enrich(tg, re, cached_ae, cached_w, cached_bi, base_scorer)
        e0 = torch.sigmoid(base_scorer.score(hg, re, tg0) - self.tau)
        f0 = base_alpha * s0 + (1 - base_alpha) * e0
        ss, es, fs = [s0], [e0], [f0]
        ps, pe, ptg = s0, e0, tg0

        # Iterative refinement with DB optimizations
        for r in range(1, self.n_rounds):
            delta = torch.abs(ps - pe)

            # Eddy gate: full early termination
            if delta.mean().item() < self.eddy_eps:
                for _r in range(r, self.n_rounds):
                    fs.append(fs[-1]); ss.append(ss[-1]); es.append(es[-1])
                break

            # Progressive epsilon narrowing
            eps = self.delta_eps[r - 1]
            active = delta > eps
            any_active = active.any().item()
            if not any_active:
                for _r in range(r, self.n_rounds):
                    fs.append(fs[-1]); ss.append(ss[-1]); es.append(es[-1])
                break

            df = self.delta_mod(delta.unsqueeze(-1)).squeeze(-1)
            sr = self._struct_score(cached_w, delta_factor=df)
            tgr = self._enrich(tg, re, cached_ae, cached_w, cached_bi, base_scorer, delta=delta)
            er = torch.sigmoid(base_scorer.score(hg, re, tgr) - self.tau)
            ar = self.compute_alpha(deg_tensor, delta=delta, round_idx=r)
            fr = ar * sr + (1 - ar) * er

            # Progressive narrowing: converged entities keep previous
            if eps > 0 and active.sum() < B:
                fr = torch.where(active, fr, fs[-1])
                sr = torch.where(active, sr, ss[-1])
                er = torch.where(active, er, es[-1])

            ss.append(sr); es.append(er); fs.append(fr)
            ps, pe, ptg = sr, er, tgr

        stacked = torch.stack(fs, dim=-1)
        rw = torch.softmax(self.round_weights, dim=0)
        S_fused = (stacked * rw).sum(dim=-1)
        return S_fused, ss[-1], es[-1], base_alpha


class DAGeR_v13(nn.Module):
    def __init__(self, ne, nr, dim=128, max_k=8, dropout=0.1,
                 lambda_bridge=0.2, scorer='transe', n_rounds=3):
        super().__init__()
        self.ne = ne; self.nr = nr; self.dim = dim
        self.max_k = max_k; self.lambda_bridge = lambda_bridge
        self.scorer_name = scorer
        if scorer not in SCORER_REGISTRY:
            raise ValueError(f"Unknown scorer '{scorer}'. Available: {list(SCORER_REGISTRY.keys())}")
        self.scorer = SCORER_REGISTRY[scorer](ne, nr, dim, dropout)
        
        self.enhance = StructuralEnhancement(dim, max_k, dropout,
                                             lambda_bridge, n_rounds=n_rounds)
    def set_bridge(self, b): self.enhance.set_bridge(b)
    def get_alpha(self): return self.enhance.get_alpha()
    def get_alpha_for_degree(self, d): return self.enhance.get_alpha_for_degree(d)
    @property
    def ent(self): return self.scorer.ent
    @property
    def rel(self): return self.scorer.rel
    def forward(self, hi, ri, ti, raw_head=None, raw_tail=None, raw_rel=None, tail_deg=None):
        he, re, te = self.scorer.encode(hi, ri, ti)
        return self.enhance(he, re, te, self.scorer, raw_head, raw_tail, raw_rel, tail_deg)


def train_epoch(model, loader, opt, device, beta=0.02, epoch=0, total_epochs=30):
    model.train()
    total_loss = 0.0; n = 0
    for batch in tqdm(loader, desc="Train"):
        hi = batch['head'].to(device); ri = batch['relation'].to(device)
        ti = batch['tail'].to(device); lab = batch['label'].to(device)
        rh = batch['raw_head'].to(device); rt = batch['raw_tail'].to(device)
        rr = batch['raw_rel'].to(device); td = batch['tail_degree'].to(device)
        lt_mask = batch['is_long_tail'].to(device)
        S_fused, S_struct, S_emb, alpha = model(hi, ri, ti,
            raw_head=rh, raw_tail=rt, raw_rel=rr, tail_deg=td)
        bce = F.binary_cross_entropy(S_fused, lab)
        hm = (lt_mask == 0).float()
        align = ((S_struct - S_emb)**2 * hm).sum() / max(hm.sum(), 1) if hm.sum() > 0 else 0.0
        alpha_t = model.enhance.compute_alpha(torch.tensor([1.0], device=device))
        alpha_h = model.enhance.compute_alpha(torch.tensor([10.0], device=device))
        alpha_avg = (alpha_t + alpha_h) / 2
        ent = -(alpha_avg * torch.log(alpha_avg+1e-8) + (1-alpha_avg)*torch.log(1-alpha_avg+1e-8))
        loss = bce + 0.05 * align - beta * ent
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        total_loss += loss.item(); n += 1
        for v in [S_fused, S_struct, S_emb, alpha, bce, loss, hi, ri, ti, lab, rh, rt, rr, td, lt_mask]:
            del v
        if n % 500 == 0: torch.cuda.empty_cache()
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    P, L, T = [], [], []
    for batch in tqdm(loader, desc="Eval"):
        hi = batch['head'].to(device); ri = batch['relation'].to(device)
        ti = batch['tail'].to(device); lab = batch['label'].to(device)
        rh = batch['raw_head'].to(device); rt = batch['raw_tail'].to(device)
        rr = batch['raw_rel'].to(device); lt = batch['is_long_tail'].to(device)
        td = batch['tail_degree'].to(device)
        S_fused, _, _, _ = model(hi, ri, ti, raw_head=rh, raw_tail=rt, raw_rel=rr, tail_deg=td)
        P.append(S_fused.detach().cpu()); L.append(lab.cpu()); T.append(lt.cpu())
        for v in [S_fused, hi, ri, ti, lab, rh, rt, rr, lt, td]: del v
        if len(P) % 500 == 0: torch.cuda.empty_cache()
    P = torch.cat(P).numpy(); L = torch.cat(L).numpy(); T = torch.cat(T).numpy()
    BP = (P >= 0.5).astype(np.float32); C = (BP == L)
    acc = C.mean()
    tp = ((BP==1)&(L==1)).sum(); fp = ((BP==1)&(L==0)).sum(); fn = ((BP==0)&(L==1)).sum()
    pr = tp/max(tp+fp,1); rc = tp/max(tp+fn,1); f1 = 2*pr*rc/max(pr+rc,1e-8)
    r = {'acc':acc,'prec':pr,'rec':rc,'f1':f1,'n':len(L)}
    for label, mask in [('lt',T==1),('hd',T==0)]:
        if mask.sum()>0:
            bm=BP[mask]; lm=L[mask]; cm=(bm==lm)
            tp_=((bm==1)&(lm==1)).sum(); fp_=((bm==1)&(lm==0)).sum(); fn_=((bm==0)&(lm==1)).sum()
            p_=tp_/max(tp_+fp_,1); r_=tp_/max(tp_+fn_,1)
            r[f'{label}_acc']=cm.mean(); r[f'{label}_f1']=2*p_*r_/max(p_+r_,1e-8); r[f'{label}_n']=int(mask.sum())
    return r


def run(name, data_dir, out_dir, dim=128, max_k=16, bs=512, epochs=40,
        lr=0.001, seed=2025, lambda_bridge=0.2, beta_entropy=0.01,
        expansion_thresh=2, ablation=None, scorer='transe', n_rounds=3,
        bert_path=None):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"DAGeR v13 | {name} | dim={dim} max_k={max_k} n_rounds={n_rounds}")
    print(f"  DB-Optimized Iterative Feedback | ablation={ablation or 'none'}")
    print(f"{'='*60}")
    t0 = time.time()
    t1 = read_triples(os.path.join(data_dir, 'triples_1'))
    t2 = read_triples(os.path.join(data_dir, 'triples_2'))
    e2i, r2i = build_ent_rel_mappings(data_dir)
    idx2raw = {v: k for k, v in e2i.items()}
    deg = compute_node_degrees(os.path.join(data_dir, 'triples_1'),
                               os.path.join(data_dir, 'triples_2'))
    print(f"  Data: {len(e2i)}e {len(r2i)}r | load={time.time()-t0:.1f}s")
    t_pre = time.time()

    print(f"\n  --- [M1] IDFWeightedSignatures ---")
    t1s = time.time()
    fusion_train = read_triples(os.path.join(data_dir, 'sup_triples_1_fusion'))
    train_entities = set()
    for h, r, t, _, _ in fusion_train:
        train_entities.add(h); train_entities.add(t)
    train_entities.discard(-1)
    sig = IDFWeightedSignatures(expansion_thresh=expansion_thresh)
    sig.build(t1)
    print(f"  [M1] {time.time()-t1s:.1f}s")

    print(f"\n  --- [M2] ParametricSimulation ---")
    t2s = time.time()
    sim = ParametricSimulation(sig, max_k=max_k)
    sim.build(fusion_train, e2i, deg)
    print(f"  [M2] {time.time()-t2s:.1f}s")

    print(f"\n  --- [M3] MultiConstraintBridge ---")
    t3s = time.time()
    bridge = MultiConstraintBridge(sig, idx2raw)
    bridge.build(sim, deg)
    print(f"  [M3] {time.time()-t3s:.1f}s")

    print(f"\n  --- Negative Sampling ---")
    tns = time.time()
    all_ents = set(list(e2i.keys()))
    neg_sampler = NegativeSampler(fusion_train, all_ents, num_neg=1, seed=seed)
    neg_triples = neg_sampler.generate(fusion_train)
    print(f"  [Neg] {len(neg_triples)} negatives in {time.time()-tns:.1f}s")
    t_pre_total = time.time() - t_pre
    print(f"\n  Phase I: {t_pre_total:.1f}s")

    all_train = fusion_train + neg_triples
    all_labels = [1] * len(fusion_train) + [0] * len(neg_triples)
    vp = read_triples(os.path.join(data_dir, 'ref_triples_1_fusion'))
    vn = read_triples(os.path.join(data_dir, 'false_ref_triples_1_fusion'))
    tr_ds = FusionDataset(all_train, all_labels, e2i, r2i, deg)
    te_ds = FusionDataset(vp + vn, [1]*len(vp)+[0]*len(vn), e2i, r2i, deg)
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn,
                       num_workers=0, pin_memory=True)
    te_ld = DataLoader(te_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn,
                       num_workers=0, pin_memory=True)
    lttr = sum(1 for d in tr_ds.data if d[8])
    ltte = sum(1 for d in te_ds.data if d[8])
    print(f"  Train={len(tr_ds)} (LT={lttr}) Test={len(te_ds)} (LT={ltte})")

    if ablation == 'emb_only': lambda_bridge = 0.0
    model = DAGeR_v13(len(e2i), len(r2i), dim=dim, max_k=max_k,
                      lambda_bridge=lambda_bridge, scorer=scorer,
                      n_rounds=n_rounds).to(dev)
    if ablation in ('no_bridge',): model.set_bridge(None)
    else: model.set_bridge(bridge)
    if scorer == 'bert':
        bp = bert_path or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                       'pretrained_models', 'bert-base-uncased')
        init_bert_scorer_weights(model.scorer, data_dir, e2i, r2i, bp, device=dev)
    n_params = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [M4] {n_params:,} params ({n_tr} trainable) | n_rounds={n_rounds}")
    print(f"  DB opts: cached_ae, delta_gating, progressive_eps")

    pos_ratio = len(fusion_train) / max(len(all_train), 1)
    with torch.no_grad():
        if 0 < pos_ratio < 1:
            model.enhance.tau.data = torch.tensor(-math.log(pos_ratio/(1-pos_ratio)))
    print(f"  tau={model.enhance.tau.item():.4f} pos_ratio={pos_ratio:.3f}")
    print(f"  alpha tail={model.get_alpha_for_degree(1.0):.4f} head={model.get_alpha_for_degree(10.0):.4f}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_res, times = 0, None, []
    for ep in range(epochs):
        t0_ep = time.time()
        if dev == 'cuda': torch.cuda.empty_cache()
        loss = train_epoch(model, tr_ld, opt, dev, beta=beta_entropy, epoch=ep, total_epochs=epochs)
        tr_t = time.time() - t0_ep
        t0_ev = time.time()
        if dev == 'cuda': torch.cuda.empty_cache()
        res = evaluate(model, te_ld, dev)
        ev_t = time.time() - t0_ev
        times.append(tr_t + ev_t)
        s = "*" if res['f1'] > best else " "
        if res['f1'] > best:
            best = res['f1']; best_res = dict(res)
            best_res['tau'] = model.enhance.tau.item()
            best_res['alpha_tail'] = model.get_alpha_for_degree(1.0)
            best_res['alpha_head'] = model.get_alpha_for_degree(10.0)
        ls = f"LT F1={res.get('lt_f1',0):.4f}" if 'lt_f1' in res else ""
        at = model.get_alpha_for_degree(1.0); ah = model.get_alpha_for_degree(10.0)
        print(f"  E{ep+1:3d}{s} ACC={res['acc']:.4f} F1={res['f1']:.4f} {ls} "
              f"| tr={tr_t:.1f}s ev={ev_t:.1f}s loss={loss:.4f} "
              f"tau={model.enhance.tau.item():.3f} a_tail={at:.3f} a_head={ah:.3f}")

    avg_t = sum(times)/max(len(times),1)
    print(f"\n  Avg epoch: {avg_t:.1f}s | Total: {sum(times):.1f}s")
    if torch.cuda.is_available():
        print(f"  GPU mem: alloc={torch.cuda.max_memory_allocated()/1024**2:.0f}MiB "
              f"reserved={torch.cuda.max_memory_reserved()/1024**2:.0f}MiB")
    print(f"\n{'='*60}")
    print(f"  BEST (tau={best_res.get('tau',0):.4f}):")
    print(f"  ACC={best_res['acc']:.4f} F1={best_res['f1']:.4f}")
    for k in ['lt','hd']:
        if f'{k}_f1' in best_res:
            print(f"  {k.upper()}: ACC={best_res[f'{k}_acc']:.4f} F1={best_res[f'{k}_f1']:.4f} (n={best_res[f'{k}_n']})")
    print(f"{'='*60}\n")
    res_path = os.path.join(out_dir, f"results_{name}.txt")
    with open(res_path, 'w') as f:
        f.write(f"DAGeR v13 | {name} | dim={dim} max_k={max_k} n_rounds={n_rounds}\n")
        f.write(f"BEST: ACC={best_res['acc']:.4f} F1={best_res['f1']:.4f} tau={best_res.get('tau',0):.4f}\n")
        for k in ['lt','hd']:
            if f'{k}_f1' in best_res:
                f.write(f"{k.upper()}: ACC={best_res[f'{k}_acc']:.4f} F1={best_res[f'{k}_f1']:.4f} (n={best_res[f'{k}_n']})\n")
        f.write(f"Phase I: {t_pre_total:.1f}s | Avg epoch: {avg_t:.1f}s\n")
    ckpt_path = os.path.join(out_dir, f"model_{name}.pt")
    torch.save({'model_state_dict': model.state_dict(), 'opt_state_dict': opt.state_dict(),
                'best_res': best_res, 'args': {'dim':dim,'max_k':max_k,'lambda_bridge':lambda_bridge,
                'n_rounds':n_rounds, 'version':'v13'}}, ckpt_path)
    print(f"  Saved: {res_path}"); print(f"  Saved: {ckpt_path}")
    return best_res, avg_t


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='DAGeR v13: DB-Optimized Iterative Feedback')
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--name', type=str, default='dataset')
    p.add_argument('--output_dir', type=str, default='./outputs')
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--max_k', type=int, default=16)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--lambda_bridge', type=float, default=0.2)
    p.add_argument('--beta_entropy', type=float, default=0.01)
    p.add_argument('--expansion_thresh', type=int, default=2)
    p.add_argument('--scorer', type=str, default='transe',
                   choices=list(SCORER_REGISTRY.keys()))
    p.add_argument('--n_rounds', type=int, default=3)
    p.add_argument('--ablation', type=str, default=None, choices=['no_bridge','emb_only'])
    p.add_argument('--bert_path', type=str, default=None)
    args = p.parse_args()
    run(args.name, args.data_dir, args.output_dir, dim=args.dim, max_k=args.max_k,
        bs=args.batch_size, epochs=args.epochs, lr=args.lr, seed=args.seed,
        lambda_bridge=args.lambda_bridge, beta_entropy=args.beta_entropy,
        expansion_thresh=args.expansion_thresh, ablation=args.ablation,
        scorer=args.scorer, n_rounds=args.n_rounds, bert_path=args.bert_path)
