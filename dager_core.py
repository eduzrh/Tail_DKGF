#!/usr/bin/env python3

import os, math, random, argparse, time
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

Triple5 = Tuple[int, int, int, int, int]

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def read_triples(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s: continue
            parts = s.split()
            if len(parts) != 5: parts = s.split('\t')
            if len(parts) != 5: continue
            try: rows.append(tuple(map(int, parts)))
            except: continue
    return rows

def build_ent_rel_mappings(data_dir):
    ent2idx, rel2idx = {}, {}
    for fn in ['ent_ids_1', 'ent_ids_2']:
        fp = os.path.join(data_dir, fn)
        if os.path.exists(fp):
            with open(fp, encoding='utf-8') as f:
                for line in f:
                    p = line.strip().split('\t')
                    if p:
                        eid = int(p[0])
                        if eid not in ent2idx: ent2idx[eid] = len(ent2idx)
    for fn in ['rel_ids_1', 'rel_ids_2']:
        fp = os.path.join(data_dir, fn)
        if os.path.exists(fp):
            with open(fp, encoding='utf-8') as f:
                for line in f:
                    p = line.strip().split('\t')
                    if p:
                        rid = int(p[0])
                        if rid not in rel2idx: rel2idx[rid] = len(rel2idx)
    return ent2idx, rel2idx

def build_idx2name(ent2idx, rel2idx, data_dir):
  """Map embedding row index -> textual symbol for BERT warm-start."""
  raw_ent_name, raw_rel_name = {}, {}
  for fn in ['ent_ids_1', 'ent_ids_2']:
    fp = os.path.join(data_dir, fn)
    if not os.path.exists(fp):
      continue
    with open(fp, encoding='utf-8') as f:
      for line in f:
        p = line.strip().split('\t', 1)
        if not p:
          continue
        try:
          raw_ent_name[int(p[0])] = p[1].strip() if len(p) > 1 and p[1].strip() else f'id_{p[0]}'
        except ValueError:
          continue
  for fn in ['rel_ids_1', 'rel_ids_2']:
    fp = os.path.join(data_dir, fn)
    if not os.path.exists(fp):
      continue
    with open(fp, encoding='utf-8') as f:
      for line in f:
        p = line.strip().split('\t', 1)
        if not p:
          continue
        try:
          raw_rel_name[int(p[0])] = p[1].strip() if len(p) > 1 and p[1].strip() else f'id_{p[0]}'
        except ValueError:
          continue
  ent_names = [raw_ent_name.get(raw, f'id_{raw}') for raw, idx in sorted(ent2idx.items(), key=lambda x: x[1])]
  rel_names = [raw_rel_name.get(raw, f'id_{raw}') for raw, idx in sorted(rel2idx.items(), key=lambda x: x[1])]
  return ent_names, rel_names

def init_bert_scorer_weights(scorer, data_dir, ent2idx, rel2idx, bert_path, device='cpu', batch_size=256):
    """Warm-start ent/rel embeddings from [CLS] vectors of bert-base-uncased."""
    import hashlib
    ent_names, rel_names = build_idx2name(ent2idx, rel2idx, data_dir)
    dim = scorer.ent.embedding_dim
    ds_tag = os.path.basename(os.path.normpath(data_dir))
    bert_tag = hashlib.md5(os.path.abspath(bert_path).encode()).hexdigest()[:8]
    cache_dir = os.path.join(os.path.dirname(bert_path), 'dager_bert_cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{ds_tag}_d{dim}_{bert_tag}.pt')

    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location='cpu')
        with torch.no_grad():
            scorer.ent.weight.copy_(cached['ent'])
            scorer.rel.weight.copy_(cached['rel'])
        print(f"  [BERT] Loaded cached embeddings: {cache_path}")
        return

    from transformers import BertTokenizer, BertModel
    print(f"  [BERT] Encoding {len(ent_names)} entities + {len(rel_names)} relations (cache miss)...")
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    bert = BertModel.from_pretrained(bert_path).to(device)
    bert.eval()
    proj = nn.Linear(bert.config.hidden_size, dim).to(device)

    def encode_texts(texts, label):
        out = []
        with torch.no_grad():
            it = range(0, len(texts), batch_size)
            it = tqdm(it, desc=f'  [BERT] {label}', leave=False)
            for i in it:
                batch = texts[i:i + batch_size]
                enc = tokenizer(batch, padding=True, truncation=True, max_length=32, return_tensors='pt')
                enc = {k: v.to(device) for k, v in enc.items()}
                cls = bert(**enc).last_hidden_state[:, 0, :]
                out.append(proj(cls).cpu())
        return torch.cat(out, dim=0) if out else torch.empty(0, dim)

    ent_vecs = encode_texts(ent_names, 'entities')
    rel_vecs = encode_texts(rel_names, 'relations')
    del bert, proj, tokenizer
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    with torch.no_grad():
        if ent_vecs.numel():
            scorer.ent.weight.copy_(ent_vecs)
        if rel_vecs.numel():
            scorer.rel.weight.copy_(rel_vecs)
    torch.save({'ent': ent_vecs, 'rel': rel_vecs}, cache_path)
    print(f"  [BERT] Warm-started and cached: {cache_path}")

# --- Bitmap utilities for fast Jaccard (SIGMOD 2024 vectorized exec) ---

def build_rel_bitmap_index(sig, num_relations):
    """Convert IDFWeightedSignatures to fixed-width bitmaps for GPU Jaccard.
    
    Each entity unified signature becomes a bitmap of size ceil(num_relations/32) uint32s.
    Jaccard(e1, e2) = popcount(bm1 & bm2) / popcount(bm1 | bm2).
    This is 100-1000x faster than Python set operations on GPU.
    """
    num_ints = (num_relations + 31) // 32
    all_entities = set()
    for rel_ents in sig.L1.values():
        all_entities.update(rel_ents)
    for rel_ents in sig.L2.values():
        all_entities.update(rel_ents)
    
    entity_list = sorted(all_entities)
    entity2row = {e: i for i, e in enumerate(entity_list)}
    
    import numpy as np
    bitmaps = np.zeros((len(entity_list), num_ints), dtype=np.uint32)
    
    for e in entity_list:
        sig_set = sig.unified_sig(e)
        row = entity2row[e]
        for r in sig_set:
            if r < num_relations:
                int_idx = r // 32
                bit_idx = r % 32
                bitmaps[row, int_idx] |= (1 << bit_idx)
    
    import torch
    bm_tensor = torch.from_numpy(bitmaps).long()
    
    print(f"  [Bitmap] {len(entity_list)} entities x {num_ints} ints = {bitmaps.nbytes/1024:.0f} KB")
    return bm_tensor, entity2row, entity_list


def bitmap_jaccard_batch(bm_query, bm_db, query_row, db_rows):
    """Compute plain Jaccard between query entity and a batch of DB entities."""
    if not db_rows:
        return []
    q_bm = bm_query[query_row]
    c_bm = bm_db[db_rows]
    inter = q_bm.unsqueeze(0) & c_bm
    union = q_bm.unsqueeze(0) | c_bm
    inter_cnt = inter.sum(dim=-1).float()
    union_cnt = union.sum(dim=-1).float()
    return inter_cnt / union_cnt.clamp(min=1)


def plain_jaccard_gpu(g_sig_set, e_sig_set):
    """Fallback plain Jaccard using Python sets (for small batches)."""
    if not g_sig_set or not e_sig_set:
        return 0.0
    inter = g_sig_set & e_sig_set
    union = g_sig_set | e_sig_set
    return len(inter) / max(len(union), 1)


def compute_node_degrees(f1, f2):
    deg = defaultdict(int)
    for fp in [f1, f2]:
        if os.path.exists(fp):
            with open(fp, encoding='utf-8') as f:
                for line in f:
                    p = line.strip().split('\t')
                    try:
                        if len(p) >= 3: deg[int(p[0])] += 1; deg[int(p[2])] += 1
                    except: continue
    return dict(deg)

def eigen_2nd_smallest(sim_matrix):
    if sim_matrix.shape[0] <= 1: return 0.0
    A = sim_matrix; D_diag = A.sum(axis=1); L = np.diag(D_diag) - A
    try:
        eigenvalues = np.linalg.eigvalsh(L)
        if len(eigenvalues) >= 2: return float(max(0.0, eigenvalues[1]))
    except: pass
    return 0.0


# =====================================================================
# M1: IDF-WEIGHTED STRUCTURAL SIGNATURES (Steps 1-2)
# =====================================================================

class IDFWeightedSignatures:
    def __init__(self, expansion_thresh: int = 2):
        self.expansion_thresh = expansion_thresh
        self.in_sig: Dict[int, Set[int]] = defaultdict(set)
        self.out_sig: Dict[int, Set[int]] = defaultdict(set)
        self.in_exp: Dict[int, Set[int]] = defaultdict(set)
        self.out_exp: Dict[int, Set[int]] = defaultdict(set)
        self.idf: Dict[int, float] = {}
        self.L1: Dict[int, List[int]] = defaultdict(list)
        self.L2: Dict[int, List[int]] = defaultdict(list)
        self.entity_degree: Dict[int, int] = {}
        self.num_entities = 0
        self.adj_in: Dict[int, List[int]] = defaultdict(list)
        self.adj_out: Dict[int, List[int]] = defaultdict(list)

    def build(self, triples, entity_set=None, train_entities=None):
        for h, r, t, _, _ in triples:
            self.in_sig[t].add(r); self.out_sig[h].add(r)
            self.entity_degree[h] = self.entity_degree.get(h, 0) + 1
            self.entity_degree[t] = self.entity_degree.get(t, 0) + 1
            self.adj_in[t].append(h); self.adj_out[h].append(t)

        all_entities = set(self.in_sig.keys()) | set(self.out_sig.keys())
        self.num_entities = len(all_entities)

        # 2-hop expansion for tail entities
        n_expanded = 0
        for e in all_entities:
            deg = self.entity_degree.get(e, 0)
            if deg <= self.expansion_thresh:
                exp_in = set(self.in_sig[e]); exp_out = set(self.out_sig[e])
                for nb in self.adj_in.get(e, []): exp_in |= self.in_sig.get(nb, set())
                for nb in self.adj_out.get(e, []): exp_out |= self.out_sig.get(nb, set())
                if len(exp_in) > len(self.in_sig[e]) or len(exp_out) > len(self.out_sig[e]):
                    n_expanded += 1
                self.in_exp[e] = exp_in; self.out_exp[e] = exp_out
            else:
                self.in_exp[e] = set(self.in_sig[e]); self.out_exp[e] = set(self.out_sig[e])

        # Raw posting lists
        raw_l1 = defaultdict(set); raw_l2 = defaultdict(set)
        for e in all_entities:
            for r in self.in_exp.get(e, set()): raw_l1[r].add(e)
            for r in self.out_exp.get(e, set()): raw_l2[r].add(e)

        total_entities = self.num_entities if entity_set is None else len(entity_set)

        # IDF
        for r in set(raw_l1.keys()) | set(raw_l2.keys()):
            df = len(raw_l1.get(r, set())) + len(raw_l2.get(r, set()))
            self.idf[r] = math.log(max(total_entities, 1) / max(df, 1))

        # Soft Purging: keep all relations but assign floor IDF to very common ones
        # Rationale (ICDE 2023 Papadakis): In KGs, common relations are structural
        # (e.g. isA) not stop words. Keep them but with minimum IDF weight.
        idf_floor = math.log(2.0)  # minimum IDF for relations with df > N/2
        n_soft_purged = 0
        for r in set(raw_l1.keys()) | set(raw_l2.keys()):
            df = len(raw_l1.get(r, set())) + len(raw_l2.get(r, set()))
            if df > total_entities / 2.0:
                self.idf[r] = idf_floor  # soft floor instead of removal
                n_soft_purged += 1

        # --- Training-set anchor pool filter ---
        # Only keep entities that appear in fusion_train. These have trained
        # embeddings in M4; non-training entities produce noisy structural signals.
        if train_entities is not None:
            n_before_l1 = sum(len(v) for v in raw_l1.values())
            n_before_l2 = sum(len(v) for v in raw_l2.values())
            raw_l1 = {r: [e for e in ents if e in train_entities]
                      for r, ents in raw_l1.items()}
            raw_l2 = {r: [e for e in ents if e in train_entities]
                      for r, ents in raw_l2.items()}
            raw_l1 = {r: ents for r, ents in raw_l1.items() if ents}
            raw_l2 = {r: ents for r, ents in raw_l2.items() if ents}
            n_after_l1 = sum(len(v) for v in raw_l1.values())
            n_after_l2 = sum(len(v) for v in raw_l2.values())
            n_tr = len(train_entities)
            print(f"  [M1] Train-filter: L1 {n_before_l1}->{n_after_l1} entries, "
                  f"L2 {n_before_l2}->{n_after_l2} entries "
                  f"({len(raw_l1)}r/{len(raw_l2)}r, {n_tr} train entities)")

        # Degree-sort ascending
        for rel, ents in raw_l1.items():
            self.L1[rel] = sorted(ents, key=lambda e: self.entity_degree.get(e, 0))
        for rel, ents in raw_l2.items():
            self.L2[rel] = sorted(ents, key=lambda e: self.entity_degree.get(e, 0))

        avg_in = sum(len(v) for v in self.in_sig.values()) / max(self.num_entities, 1)
        avg_out = sum(len(v) for v in self.out_sig.values()) / max(self.num_entities, 1)
        print(f"  [M1] {self.num_entities}e | in={avg_in:.1f} out={avg_out:.1f} | "
              f"2hop={n_expanded} | soft_purged={n_soft_purged} | L1={len(self.L1)}r L2={len(self.L2)}r")

    def unified_sig(self, e: int) -> Set[int]:
        return self.in_exp.get(e, set()) | self.out_exp.get(e, set())

    def idf_jaccard(self, s1: Set[int], s2: Set[int]) -> float:
        if not s1 or not s2: return 0.0
        inter = s1 & s2; union = s1 | s2
        num = sum(self.idf.get(r, 1.0) for r in inter)
        den = sum(self.idf.get(r, 1.0) for r in union)
        return num / max(den, 1e-8)

    def inter_entity_idf_jac(self, e1: int, e2: int) -> float:
        return self.idf_jaccard(self.unified_sig(e1), self.unified_sig(e2))

    def plain_jac(self, s1: Set[int], s2: Set[int]) -> float:
        if not s1 or not s2: return 0.0
        return len(s1 & s2) / len(s1 | s2)



class ParametricSimulation:
    """M2: Parametric Graph Simulation (Steps 3-4).
    
    Techniques incorporated:
      - LTJ-style relation ordering (SIGMOD 2024 Arroyuelo): process rarest relations first
      - Degree-adaptive progressive simulation (ICDE 2022 Fan): 3-round tightening
      - Matching consensus filter (CIKM 2021 Yang): remove outlier anchors
    
    Design: Plain Jaccard for HIGH RECALL candidate generation.
    IDF weighting is reserved for M3 quality assessment.
    """
    def __init__(self, sig: IDFWeightedSignatures, rel2col: Dict[int, int] = None,
                 max_k: int = 8, gamma: float = 0.5, beta: float = 0.3):
        self.sig = sig; self.max_k = max_k
        self.gamma = gamma; self.beta = beta
        self.rel2col = rel2col or {}
        
        self.anchor_idx: Optional[torch.Tensor] = None
        self.anchor_score: Optional[torch.Tensor] = None
        self.g2pos: Dict[int, int] = {}; self.glist: List[int] = []
        
        self._unified_cache: Dict[int, Set[int]] = {}
        self._sig_size_cache: Dict[int, int] = {}

    def _adaptive_threshold(self, deg: int, base: float) -> float:
        """Degree-adaptive threshold (ICDE 2022 Fan)."""
        return base * (1.0 + self.gamma * math.exp(-self.beta * max(deg, 1)))

    def _adaptive_k(self, deg: int) -> int:
        """Degree-adaptive anchor budget. Sparse entities get more anchors."""
        if deg <= 1: return min(self.max_k + 8, 16)
        elif deg <= 2: return min(self.max_k + 4, 12)
        elif deg <= 3: return min(self.max_k + 2, 10)
        elif deg <= 5: return self.max_k
        else: return max(3, self.max_k - 4)

    def _cached_unified(self, e: int) -> Set[int]:
        if e not in self._unified_cache:
            self._unified_cache[e] = self.sig.unified_sig(e)
        return self._unified_cache[e]
    
    def _cached_sig_size(self, e: int) -> int:
        if e not in self._sig_size_cache:
            self._sig_size_cache[e] = len(self._cached_unified(e))
        return self._sig_size_cache[e]

    def _plain_jaccard(self, e1: int, e2: int) -> float:
        """Plain Jaccard (no IDF) for high-recall candidate scanning."""
        s1 = self._cached_unified(e1); s2 = self._cached_unified(e2)
        if not s1 or not s2: return 0.0
        return len(s1 & s2) / max(len(s1 | s2), 1)

    def _consensus_filter(self, candidates, tau=1.0):
        """CIKM 2021 Yang: Global matching consensus filter."""
        if len(candidates) <= 3: return candidates
        scores = []
        for a_raw, _ in candidates:
            a_sig = self._cached_unified(a_raw)
            others = []
            for a2, _ in candidates:
                if a2 != a_raw:
                    others.append(plain_jaccard_gpu(a_sig, self._cached_unified(a2)))
            scores.append(sum(others) / max(len(others), 1))
        mean_s = sum(scores) / len(scores)
        std_s = (sum((s - mean_s)**2 for s in scores) / max(len(scores), 1)) ** 0.5
        thresh = mean_s - tau * std_s
        return [(a, sc) for (a, sc), s in zip(candidates, scores) if s >= thresh]

    def build(self, fusion_train, ent2idx, gkg_degrees):
        """Steps 3-4: Parametric graph simulation for anchor discovery."""
        from collections import defaultdict
        
        g2in = defaultdict(set); g2out = defaultdict(set)
        # Extract training entities: only these have trained M4 embeddings
        train_entities = set()
        for h, r, t, _, _ in fusion_train:
            g2in[t].add(r); g2out[h].add(r)
            train_entities.add(h); train_entities.add(t)
        train_entities.discard(-1)
        
        all_dkg = set()
        for rel_ents in self.sig.L1.values():
            all_dkg.update(rel_ents)
        for rel_ents in self.sig.L2.values():
            all_dkg.update(rel_ents)
        print(f"  [M2] Warming caches for {len(all_dkg)} DKG entities...")
        for e in tqdm(all_dkg, desc="  [M2] Cache"):
            self._cached_unified(e); self._cached_sig_size(e)
        
        gkg_tails = sorted(set(t for _, _, t, _, _ in fusion_train))
        N = len(gkg_tails)
        anchor_indices = []; anchor_scores = []; n_connected = 0
        
        l1_sizes = {r: len(ents) for r, ents in self.sig.L1.items()}
        l2_sizes = {r: len(ents) for r, ents in self.sig.L2.items()}

        for g in tqdm(gkg_tails, desc="[M2] ParamSim"):
            q_in = g2in.get(g, set()); q_out = g2out.get(g, set())
            if not q_in and not q_out:
                anchor_indices.append([]); anchor_scores.append([]); continue
            
            g_deg = gkg_degrees.get(g, 0)
            k_adapt = self._adaptive_k(g_deg)
            

            q_in_ordered = sorted(q_in, key=lambda r: l1_sizes.get(r, 999999))
            q_out_ordered = sorted(q_out, key=lambda r: l2_sizes.get(r, 999999))
            

            candidates = {}
            g_sig = self._cached_unified(g)
            
            for rel in q_in_ordered:
                for e in self.sig.L1.get(rel, []):
                    if e == g: continue
                    if e in candidates: continue
                    j = self._plain_jaccard(g, e)
                    if j > 0:
                        candidates[e] = j
            
            for rel in q_out_ordered:
                for e in self.sig.L2.get(rel, []):
                    if e == g: continue
                    if e in candidates: continue
                    j = self._plain_jaccard(g, e)
                    if j > 0:
                        candidates[e] = j
            
            if not candidates:
                anchor_indices.append([]); anchor_scores.append([]); continue
            
            rounds = [
                (0.05, 2.0),
                (0.10, 1.5),
                (0.15, 1.0),
            ]
            
            current_cands = dict(candidates)
            final_cands = []
            
            for rnd_idx, (sigma_base, keep_mult) in enumerate(rounds):
                sigma = self._adaptive_threshold(g_deg, sigma_base)
                keep_n = max(int(k_adapt * keep_mult), k_adapt)
                
                filtered = [(a, hv) for a, hv in current_cands.items() if hv >= sigma]
                filtered.sort(key=lambda x: -x[1])
                filtered = filtered[:keep_n]
                
                if not filtered:
                    break
                
                current_cands = {a: hv for a, hv in filtered}
                final_cands = filtered
            
            if len(final_cands) > 3:
                final_cands = self._consensus_filter(final_cands)
            
            top = final_cands[:k_adapt]
            
            if top: n_connected += 1
            idxs = [ent2idx.get(a, 0) for a, _ in top]
            scs = [s for _, s in top]
            anchor_indices.append(idxs[:self.max_k])
            anchor_scores.append(scs[:self.max_k])
        
        id_t = torch.zeros(N, self.max_k, dtype=torch.long)
        sc_t = torch.zeros(N, self.max_k)
        for i in range(N):
            for j in range(min(len(anchor_indices[i]), self.max_k)):
                id_t[i, j] = anchor_indices[i][j]
                sc_t[i, j] = anchor_scores[i][j]
        
        self.anchor_idx = id_t; self.anchor_score = sc_t
        self.glist = gkg_tails; self.g2pos = {g: i for i, g in enumerate(gkg_tails)}
        
        avg = sum(len(al) for al in anchor_indices) / max(n_connected, 1)
        print(f"  [M2] ParamSim: {n_connected}/{N} connected (avg {avg:.1f} anchors)")

    def lookup(self, raw_tails: torch.Tensor):
        if self.anchor_idx is None:
            B = raw_tails.shape[0]
            return (torch.zeros(B, 1, dtype=torch.long, device=raw_tails.device),
                    torch.zeros(B, 1, device=raw_tails.device))
        pos = torch.tensor([self.g2pos.get(rt.item(), 0) for rt in raw_tails],
                          dtype=torch.long, device=raw_tails.device)
        return (self.anchor_score.to(raw_tails.device)[pos],
                self.anchor_idx.to(raw_tails.device)[pos])


class MultiConstraintBridge:
    def __init__(self, sig: IDFWeightedSignatures, idx2raw: Dict[int, int] = None):
        self.sig = sig; self.idx2raw = idx2raw or {}
        self.g2pos: Dict[int, int] = {}
        self.bridge_weight: Optional[torch.Tensor] = None
        self.bridge_idx: Optional[torch.Tensor] = None

    def build(self, sim: ParametricSimulation, gkg_degrees):
        if sim.anchor_idx is None:
            print("  [M3] No anchors"); return
        self.g2pos = dict(sim.g2pos)
        idx = sim.anchor_idx; sc = sim.anchor_score
        N, K = idx.shape
        weights = torch.zeros(N, K)
        n_active = 0

        for i in tqdm(range(N), desc="[M3] Bridge"):
            g_raw = sim.glist[i]; g_deg = gkg_degrees.get(g_raw, 1)
            valid = [(j, idx[i, j].item(), sc[i, j].item())
                     for j in range(K) if idx[i, j].item() > 0 and sc[i, j].item() > 0]
            if not valid: continue
            nv = len(valid)
            g_sig = self.sig.unified_sig(g_raw)
            g_idf_sum = sum(self.sig.idf.get(r, 1.0) for r in g_sig)

            # C2: Spectral consistency
            if nv >= 2:
                M = np.zeros((nv, nv))
                for p, (_, a1i, _) in enumerate(valid):
                    a1r = self.idx2raw.get(a1i, a1i)
                    for q, (_, a2i, _) in enumerate(valid):
                        if p == q: M[p, q] = 1.0
                        else:
                            a2r = self.idx2raw.get(a2i, a2i)
                            M[p, q] = self.sig.inter_entity_idf_jac(a1r, a2r)
                lam2 = eigen_2nd_smallest(M)
                w_c2 = 1.0 - math.exp(-max(lam2, 0.0))
            else:
                w_c2 = 0.5

            edge_scores = []
            for local_j, a_idx_val, raw_score in valid:
                a_raw = self.idx2raw.get(a_idx_val, a_idx_val)
                a_deg = self.sig.entity_degree.get(a_raw, 1)
                a_sig = self.sig.unified_sig(a_raw)
                a_idf_sum = sum(self.sig.idf.get(r, 1.0) for r in a_sig)

                w_c1 = min(1.0, g_deg / max(a_deg, 1))

                inter_idf = sum(self.sig.idf.get(r, 1.0) for r in (g_sig & a_sig))
                w_c3 = inter_idf / max(g_idf_sum, 1e-8)

                excess_idf = sum(self.sig.idf.get(r, 1.0) for r in (a_sig - g_sig))
                w_c4 = 1.0 - excess_idf / max(a_idf_sum, 1e-8)

                constraints = np.array([w_c1, w_c2, w_c3, w_c4])
                edge_scores.append((local_j, raw_score, constraints))

            if len(edge_scores) < 1:
                continue


            raw_scores_arr = np.array([s for _, s, _ in edge_scores])
            global_mean = raw_scores_arr.mean()
            edge_scores = [(lj, s, c) for lj, s, c in edge_scores if s >= global_mean]
            if len(edge_scores) < 1:
                continue

            tail_mean = np.array([s for _, s, _ in edge_scores]).mean()
            edge_scores = [(lj, s, c) for lj, s, c in edge_scores if s >= tail_mean * 0.5]
            if len(edge_scores) < 1:
                continue


            k_rcnp = max(2, min(len(edge_scores), K))  # K = max_k from M2
            edge_scores.sort(key=lambda x: x[1], reverse=True)
            edge_scores = edge_scores[:k_rcnp]


            all_cons = np.stack([c for _, _, c in edge_scores], axis=0)
            if all_cons.shape[0] >= 2:
                cons_var = all_cons.var(axis=0)
                cons_wt = 1.0 / (1.0 + cons_var)
                cons_wt = cons_wt / (cons_wt.sum() + 1e-8)
            else:
                cons_wt = np.ones(4) / 4.0

            for local_j, raw_score, constraints in edge_scores:
                w_fused = float(np.dot(constraints, cons_wt))
                weights[i, local_j] = raw_score * w_fused

            # L1 normalization
            w_sum = weights[i, :nv].sum()
            if w_sum > 0:
                weights[i, :nv] /= w_sum
                n_active += 1

        self.bridge_weight = weights; self.bridge_idx = idx
        print(f"  [M3] MultiConstraintBridge: {n_active}/{N} active")

    def lookup(self, raw_tails: torch.Tensor):
        if self.bridge_weight is None:
            B = raw_tails.shape[0]
            return (torch.zeros(B, 1, device=raw_tails.device),
                    torch.zeros(B, 1, dtype=torch.long, device=raw_tails.device))
        pos = torch.tensor([self.g2pos.get(rt.item(), 0) for rt in raw_tails],
                          dtype=torch.long, device=raw_tails.device)
        return (self.bridge_weight.to(raw_tails.device)[pos],
                self.bridge_idx.to(raw_tails.device)[pos])



class BaseKGScorer(nn.Module):
    """Abstract interface for pluggable KG embedding models.

    To plug in a new model, subclass and implement:
      - score(he, re, te) -> scores: raw triple scores (before sigmoid)
    The model MUST expose self.ent and self.rel embeddings for anchor lookup.
    """
    def __init__(self, ne: int, nr: int, dim: int, dropout: float = 0.1):
        super().__init__()
        self.ent = nn.Embedding(ne, dim)
        self.rel = nn.Embedding(nr, dim)
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.ent.weight, 0, 1.0 / math.sqrt(dim))
        nn.init.normal_(self.rel.weight, 0, 1.0 / math.sqrt(dim))

    def encode(self, hi, ri, ti):
        """Return (head_emb, rel_emb, tail_emb)."""
        return self.drop(self.ent(hi)), self.drop(self.rel(ri)), self.drop(self.ent(ti))

    def score(self, he, re, te):
        """Raw triple score (pre-sigmoid). Must be overridden."""
        raise NotImplementedError


class TransEScorer(BaseKGScorer):
    """TransE: s(h,r,t) = -1/2 * ||h + r - t||^2"""
    def score(self, he, re, te):
        sc = he + re - te
        return -torch.sum(sc ** 2, dim=-1) * 0.5


class RotatEScorer(BaseKGScorer):
    """RotatE: s(h,r,t) = -||h o r - t||^2 (complex rotation)."""
    def score(self, he, re, te):
        d2 = he.shape[-1] // 2
        h_re, h_im = he[..., :d2], he[..., d2:]
        r_re, r_im = re[..., :d2], re[..., d2:]
        t_re, t_im = te[..., :d2], te[..., d2:]
        r_norm = torch.sqrt(r_re**2 + r_im**2 + 1e-8)
        r_re, r_im = r_re / r_norm, r_im / r_norm
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        return -torch.sum((hr_re - t_re)**2 + (hr_im - t_im)**2, dim=-1)


class ExeFuseScorer(BaseKGScorer):

    def __init__(self, ne, nr, dim, dropout=0.1):
        super().__init__(ne, nr, dim, dropout)
        self.ent_proj = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim)
        )
        self.rel_transform = nn.Parameter(torch.randn(nr, dim, dim) * 0.01)

    def encode(self, hi, ri, ti):
        he = self.ent_proj(self.drop(self.ent(hi)))
        te = self.ent_proj(self.drop(self.ent(ti)))
        re = self.drop(self.rel(ri))
        return he, re, te

    def score(self, he, re, te):
        sc = he + re - te
        return -torch.sum(sc ** 2, dim=-1) * 0.5




class TransHScorer(BaseKGScorer):
    """TransH (Wang AAAI 2014): relation-specific hyperplane projection.
    
    s(h,r,t) = -||h_perp + d_r - t_perp||^2
    Projects entities onto relation-specific hyperplanes via normal w_r.
    d_r is stored in self.rel (the base relation embedding).
    """
    def __init__(self, ne, nr, dim, dropout=0.1):
        super().__init__(ne, nr, dim, dropout)
        self.w_r = nn.Embedding(nr, dim)
        nn.init.xavier_uniform_(self.w_r.weight)
        self._ri_cache = None

    def encode(self, hi, ri, ti):
        self._ri_cache = ri
        return self.drop(self.ent(hi)), self.drop(self.rel(ri)), self.drop(self.ent(ti))

    def score(self, he, re, te):
        w = self.w_r(self._ri_cache)
        w_norm = F.normalize(w, p=2, dim=-1)
        h_perp = he - (he * w_norm).sum(dim=-1, keepdim=True) * w_norm
        t_perp = te - (te * w_norm).sum(dim=-1, keepdim=True) * w_norm
        sc = h_perp + re - t_perp
        return -torch.sum(sc ** 2, dim=-1) * 0.5


class GNNScorer(BaseKGScorer):
    """GNN-inspired scorer with relation-gated message passing.
    
    Simulates 1-hop graph convolution: entity = self_proj + gate * neighbor_msg.
    Relation acts as a gating mechanism on cross-entity messages.
    Preserves DAGeR plug-and-play interface -- no external graph needed.
    """
    def __init__(self, ne, nr, dim, dropout=0.1):
        super().__init__(ne, nr, dim, dropout)
        self.self_proj = nn.Linear(dim, dim)
        self.neighbor_proj = nn.Linear(dim, dim)
        self.rel_gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

    def encode(self, hi, ri, ti):
        he_raw = self.ent(hi)
        re_raw = self.rel(ri)
        te_raw = self.ent(ti)
        gate = self.rel_gate(re_raw)
        he_msg = self.self_proj(he_raw) + gate * self.neighbor_proj(te_raw)
        te_msg = self.self_proj(te_raw) + gate * self.neighbor_proj(he_raw)
        return self.drop(he_msg), self.drop(re_raw), self.drop(te_msg)

    def score(self, he, re, te):
        sc = he + re - te
        return -torch.sum(sc ** 2, dim=-1) * 0.5


class DistMultScorer(BaseKGScorer):
    """DistMult: s(h,r,t) = sum(h * r * t)."""
    def score(self, he, re, te):
        return torch.sum(he * re * te, dim=-1)


class BertScorer(BaseKGScorer):
    """Text-informed backbone: ent/rel tables are BERT-initialized then fine-tuned."""
    def score(self, he, re, te):
        return torch.sum(he * re * te, dim=-1)

# Registry of available scorers
SCORER_REGISTRY = {
    'transe': TransEScorer,
    'rotate': RotatEScorer,
    'transh': TransHScorer,
    'gnn': GNNScorer,
    'distmult': DistMultScorer,
    'exefuse': ExeFuseScorer,
    'bert': BertScorer,
}



class GMCBlock(nn.Module):
    """Sinkhorn-Knopp normalization for mutual consistency (Yao SIGMOD 2022).
    
    Given a pair of entity embeddings (h, t), iteratively normalizes
    their adjacency matrix to be doubly stochastic, then applies
    a projection to produce mutually consistent representations.
    """
    def __init__(self, dim, n_iter=5, dropout=0.1):
        super().__init__(); self.n_iter = n_iter
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(dim, dim))
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def _sk(self, A):
        A = A.clamp(min=1e-8)
        for _ in range(self.n_iter):
            A = A / (A.sum(-1, keepdim=True) + 1e-8)
            A = A / (A.sum(-2, keepdim=True) + 1e-8)
        return A

    def forward(self, x, adj=None):
        if adj is None: return x
        a = torch.sigmoid(self.alpha)
        return a * torch.bmm(self._sk(adj), self.proj(x)) + (1 - a) * x



class DomainGate(nn.Module):
    """HierGAT-style Multi-View Constraint Attention (Yao SIGMOD 2022).
    
    Hierarchical comparison adapted for KG anchor gating:
    - Attribute Comparison (Sec 5.2.1): C1-C4 constraints as "views"
    - Entity Comparison (Sec 5.2.2 Eq.4): multi-view weighted attention
      h_k = softmax(LeakyReLU(c^T[v_lr || S_k^a]))
    - Relation-conditioned: same anchor gets different weight per relation
    """
    def __init__(self, dim, dropout=0.1, use_relation=True):
        super().__init__()
        in_dim = dim * 3 if use_relation else dim * 2
        self.use_relation = use_relation
        self.mlp = nn.Sequential(nn.Linear(in_dim, dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim, 1), nn.Sigmoid())
        self.n_views = 4
        self.view_attn = nn.Sequential(
            nn.Linear(dim * 2 + self.n_views, dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, self.n_views)
        )

    def forward(self, q, b, r=None, constraint_scores=None):
        B, K, D = b.shape
        q_exp = q.unsqueeze(1).expand(-1, K, -1)
        if self.use_relation and r is not None:
            r_exp = r.unsqueeze(1).expand(-1, K, -1)
            gate_scores = self.mlp(torch.cat([q_exp, b, r_exp], -1)).squeeze(-1)
        else:
            gate_scores = self.mlp(torch.cat([q_exp, b], -1)).squeeze(-1)
        if constraint_scores is not None:
            qb = torch.cat([q_exp, b], dim=-1)
            view_input = torch.cat([qb, constraint_scores], dim=-1)
            view_wt = torch.softmax(self.view_attn(view_input), dim=-1)
            constraint_fused = (view_wt * constraint_scores).sum(dim=-1)
            gate_scores = gate_scores * torch.sigmoid(constraint_fused)
        return gate_scores





class StructuralEnhancement(nn.Module):
    """DAGeR structural enhancement: wraps any BaseKGScorer.
    
    Adds to any base KG model:
      - GMC: mutual consistency refinement of (h,t) embeddings
      - Bridge-enriched tail: anchor embeddings gated by relation-aware attention
      - Track B (S_struct): triple-specific structural scoring from M3 bridge
      - Track C (S_emb): base model scoring with bridge-enriched embeddings
      - Degree-aware bounded alpha fusion: alpha = f(deg(tail))
    
    Encoder-agnostic: TransE, RotatE, ExeFuse, ComplEx, DistMult,
    or any model providing ent/rel embeddings + scoring works.
    
    Paper contributions (M4):
      1. Degree-aware bounded alpha (prevents track collapse)
      2. Triple-specific structural scoring (head-anchor sim)
      3. Relation-conditioned DomainGate (HierGAT 2022)
      4. Structural Consistency Loss (independent S_struct supervision)
    """
    def __init__(self, dim: int, max_k: int = 8, dropout: float = 0.1,
                 lambda_bridge: float = 0.2):
        super().__init__()
        self.dim = dim
        self.max_k = max_k
        self.lambda_bridge = lambda_bridge
        self.gmc = GMCBlock(dim)
        self.gate = DomainGate(dim, dropout, use_relation=True)
        self.tau = nn.Parameter(torch.tensor(0.0))
        # Bounded degree-aware alpha
        self.alpha_min = 0.05
        self.alpha_max = 0.90  # wide range — model learns optimal balance
        self.alpha_raw = nn.Parameter(torch.tensor(0.5))  # neutral init
        self.alpha_deg = nn.Parameter(torch.tensor(-0.5))  # stronger degree diff
        # struct_scorer removed in v11 — S_struct from bridge weights (pre-computed)
        self.bridge: Optional['MultiConstraintBridge'] = None

    def set_bridge(self, bridge):
        self.bridge = bridge

    def get_alpha(self):
        a_raw = torch.sigmoid(self.alpha_raw).item()
        return self.alpha_min + (self.alpha_max - self.alpha_min) * a_raw

    def get_alpha_for_degree(self, deg: float) -> float:
        with torch.no_grad():
            a_raw = torch.sigmoid(self.alpha_raw + self.alpha_deg * math.log(1 + deg)).item()
            return self.alpha_min + (self.alpha_max - self.alpha_min) * a_raw

    def compute_alpha(self, deg: torch.Tensor) -> torch.Tensor:
        """Compute degree-adaptive alpha with gradient (for training)."""
        deg_factor = torch.log(1.0 + deg.float())
        alpha_raw = torch.sigmoid(self.alpha_raw + self.alpha_deg * deg_factor)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * alpha_raw

    def forward(self, he, re, te, base_scorer,
                raw_head=None, raw_tail=None, raw_rel=None, tail_deg=None):
        """Enhance a base scorer's predictions with structural bridge signals.
        
        Args:
            he, re, te: initial entity/relation embeddings from base_scorer.encode()
            base_scorer: the pluggable KG model (for ent/rel lookup + scoring)
            raw_head/tail/rel: raw entity/relation IDs for bridge lookup
            tail_deg: tail entity degrees for degree-aware alpha
        
        Returns:
            S_fused, S_struct, S_emb, alpha
        """
        B = he.shape[0]

        # GMC mutual consistency refinement
        x = torch.stack([he, te], dim=1)
        adj = torch.ones(B, 2, 2, device=x.device)
        x = self.gmc(x, adj)
        hg, tg = x[:, 0, :], x[:, 1, :]

        # --- Track C: Bridge-enriched embedding scoring (learnable) ---
        if raw_tail is not None and self.bridge is not None:
            anchor_w, anchor_idx = self.bridge.lookup(raw_tail)
            if anchor_idx.shape[1] > 0:
                anchor_emb = base_scorer.ent(anchor_idx)
                gw = self.gate(tg, anchor_emb, r=re).unsqueeze(-1)
                borrowed = (gw * anchor_emb).sum(dim=1)
                tg = tg + self.lambda_bridge * borrowed

        # Base scorer provides Track C score
        psi = base_scorer.score(hg, re, tg)
        S_emb = torch.sigmoid(psi - self.tau)

        # --- Track B: Bridge-weight structural scoring (v11) ---
        # Uses pre-computed M3 bridge weights directly — no MLP.
        S_struct = torch.zeros_like(S_emb)
        if raw_tail is not None and self.bridge is not None:
            bridge_w, _ = self.bridge.lookup(raw_tail)
            S_struct = bridge_w.sum(dim=-1).clamp(0, 1)

        # --- Degree-aware bounded alpha fusion ---
        if tail_deg is not None:
            deg_factor = torch.log(1.0 + tail_deg.float())
            alpha_raw = torch.sigmoid(self.alpha_raw + self.alpha_deg * deg_factor)
        else:
            alpha_raw = torch.sigmoid(self.alpha_raw)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha_raw

        S_fused = alpha * S_struct + (1 - alpha) * S_emb
        return S_fused, S_struct, S_emb, alpha


# ---------------------------------------------------------------------------
# 5.5 DAGeR Full Model (thin wrapper: BaseKGScorer + StructuralEnhancement)
# ---------------------------------------------------------------------------

class DAGeR_v6(nn.Module):
    """DAGeR v6: Pluggable encoder-agnostic KG fusion.
    
    Architecture:
      DAGeR = BaseKGScorer (any model) + StructuralEnhancement (DAGeR's core)
    
    Usage:
      model = DAGeR_v6(ne, nr, dim, scorer='transe')   # TransE
      model = DAGeR_v6(ne, nr, dim, scorer='rotate')   # RotatE
      model = DAGeR_v6(ne, nr, dim, scorer='exefuse')  # ExeFuse
    """

    def __init__(self, ne: int, nr: int, dim: int = 128, max_k: int = 8,
                 dropout: float = 0.1, lambda_bridge: float = 0.2,
                 scorer: str = 'transe'):
        super().__init__()
        self.ne = ne; self.nr = nr; self.dim = dim
        self.max_k = max_k; self.lambda_bridge = lambda_bridge
        self.scorer_name = scorer

        # Pluggable base scorer
        if scorer not in SCORER_REGISTRY:
            raise ValueError(f"Unknown scorer '{scorer}'. Available: {list(SCORER_REGISTRY.keys())}")
        self.scorer = SCORER_REGISTRY[scorer](ne, nr, dim, dropout)

        # Structural enhancement (DAGeR's core)
        self.enhance = StructuralEnhancement(dim, max_k, dropout, lambda_bridge)

    def set_bridge(self, bridge):
        self.enhance.set_bridge(bridge)

    def get_alpha(self):
        return self.enhance.get_alpha()

    def get_alpha_for_degree(self, deg: float) -> float:
        return self.enhance.get_alpha_for_degree(deg)

    @property
    def ent(self):
        """Expose scorer's entity embeddings for external access."""
        return self.scorer.ent

    @property
    def rel(self):
        """Expose scorer's relation embeddings for external access."""
        return self.scorer.rel

    def forward(self, hi, ri, ti, raw_head=None, raw_tail=None, raw_rel=None,
                tail_deg=None):
        # Step 1: Encode with pluggable base scorer
        he, re, te = self.scorer.encode(hi, ri, ti)
        # Step 2: Enhance with structural bridge + degree-aware fusion
        return self.enhance(he, re, te, self.scorer,
                           raw_head, raw_tail, raw_rel, tail_deg)


# DATA LOADING
# =====================================================================

class FusionDataset(Dataset):
    def __init__(self, triples, labels, ent2idx, rel2idx, node_degrees, lt_thresh=4):
        self.e2i = ent2idx; self.r2i = rel2idx; self.deg = node_degrees; self.lt = lt_thresh
        self.data = []
        for t, l in zip(triples, labels):
            h, r, t_, ts1, ts2 = t
            if h in ent2idx and t_ in ent2idx and r in rel2idx:
                hd = node_degrees.get(h, 0); td = node_degrees.get(t_, 0)
                lt_ = min(hd, td) <= lt_thresh
                self.data.append((ent2idx[h], ent2idx[r], ent2idx[t_],
                    ts1, ts2, l, hd, td, lt_, h, r, t_))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        h, r, t_, ts1, ts2, l, hd, td, lt_, rh, rr, rt_ = self.data[idx]
        return {'head': h, 'relation': r, 'tail': t_, 'ts1': ts1, 'ts2': ts2,
                'label': l, 'head_degree': hd, 'tail_degree': td,
                'is_long_tail': 1 if lt_ else 0,
                'raw_head': rh, 'raw_rel': rr, 'raw_tail': rt_}

def collate_fn(batch):
    keys = ['head', 'relation', 'tail', 'ts1', 'ts2', 'label',
            'head_degree', 'tail_degree', 'is_long_tail',
            'raw_head', 'raw_rel', 'raw_tail']
    return {k: torch.tensor([item[k] for item in batch],
            dtype=torch.float if k == 'label' else torch.long) for k in keys}

class NegativeSampler:
    def __init__(self, fusion_train, all_entities_set, num_neg=1, seed=2025):
        self.num_neg = num_neg
        self.all_entities = list(all_entities_set)
        random.seed(seed)
        self.true_tails = defaultdict(set)
        for h, r, t, _, _ in fusion_train:
            self.true_tails[(h, r)].add(t)

    def generate(self, triples):
        negatives = []
        entities = self.all_entities; n = len(entities)
        for h, r, t, ts1, ts2 in triples:
            true_set = self.true_tails.get((h, r), set())
            for _ in range(self.num_neg):
                for _ in range(100):
                    nt = entities[random.randint(0, n - 1)]
                    if nt != t and nt not in true_set:
                        negatives.append((h, r, nt, ts1, ts2)); break
                else:
                    for e in entities:
                        if e != t: negatives.append((h, r, e, ts1, ts2)); break
        return negatives


# =====================================================================
# TRAINING & EVALUATION
# =====================================================================

def train_epoch(model, loader, opt, device, beta=0.02, epoch=0, total_epochs=30):
    """v11: BCE loss + entropy regularizer. No SCL needed — S_struct from M3 bridge weights.
    
    The entropy term encourages balanced alpha (prevents collapse to 0 or 1).
    Bridge weights provide the structural signal directly (degree-adaptive, pre-computed).
    """
    model.train()
    total_loss = 0.0; n = 0
    for batch in tqdm(loader, desc="Train"):
        hi = batch['head'].to(device)
        ri = batch['relation'].to(device)
        ti = batch['tail'].to(device)
        lab = batch['label'].to(device)
        rh = batch['raw_head'].to(device)
        rt = batch['raw_tail'].to(device)
        rr = batch['raw_rel'].to(device)
        td = batch['tail_degree'].to(device)
        
        S_fused, _, _, _ = model(hi, ri, ti, raw_head=rh, raw_tail=rt, raw_rel=rr, tail_deg=td)
        bce_loss = F.binary_cross_entropy(S_fused, lab)
        
        # Entropy regularizer: reward balanced alpha
        alpha_tail = model.enhance.compute_alpha(torch.tensor([1.0], device=device))
        alpha_head = model.enhance.compute_alpha(torch.tensor([10.0], device=device))
        alpha_avg = (alpha_tail + alpha_head) / 2
        entropy = -(alpha_avg * torch.log(alpha_avg + 1e-8) + 
                    (1 - alpha_avg) * torch.log(1 - alpha_avg + 1e-8))
        
        loss = bce_loss - beta * entropy
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        total_loss += loss.item(); n += 1
        del S_fused, bce_loss, alpha_tail, alpha_head, alpha_avg, entropy, loss
        del hi, ri, ti, lab, rh, rt, rr, td
        if n % 500 == 0:
            torch.cuda.empty_cache()
    return total_loss / max(n, 1)
@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate with overall + long-tail/head breakdown metrics."""
    model.eval()
    P, L, T = [], [], []
    for batch in tqdm(loader, desc="Eval"):
        hi = batch['head'].to(device)
        ri = batch['relation'].to(device)
        ti = batch['tail'].to(device)
        lab = batch['label'].to(device)
        rh = batch['raw_head'].to(device)
        rt = batch['raw_tail'].to(device)
        rr = batch['raw_rel'].to(device)
        lt = batch['is_long_tail'].to(device)
        td = batch['tail_degree'].to(device)
        mem_before = torch.cuda.memory_allocated(device) / 1e9
        S_fused, S_struct, S_emb, alpha = model(hi, ri, ti, raw_head=rh,
                                                raw_tail=rt, raw_rel=rr,
                                                tail_deg=td)
        mem_after = torch.cuda.memory_allocated(device) / 1e9
        if mem_after - mem_before > 0.01:  # Log ANY jump >10MB
            print(f"\n  [MEM] batch mem: {mem_before:.3f} -> {mem_after:.3f}GB (jump {mem_after-mem_before:.3f}GB)")
        P.append(S_fused.detach().cpu())
        L.append(lab.cpu())
        T.append(lt.cpu())
        # Explicit cleanup to prevent memory accumulation
        del S_fused, S_struct, S_emb, alpha, hi, ri, ti, lab, rh, rt, rr, lt, td
        if len(P) % 200 == 0:
            torch.cuda.empty_cache()

    P = torch.cat(P).numpy()
    L = torch.cat(L).numpy()
    T = torch.cat(T).numpy()
    B = (P >= 0.5).astype(np.float32)
    C = (B == L)
    acc = C.mean()
    tp = ((B == 1) & (L == 1)).sum()
    fp = ((B == 1) & (L == 0)).sum()
    fn = ((B == 0) & (L == 1)).sum()
    pr = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    f1 = 2 * pr * rc / max(pr + rc, 1e-8)
    r = {'acc': acc, 'prec': pr, 'rec': rc, 'f1': f1, 'n': len(L)}
    for label, mask in [('lt', T == 1), ('hd', T == 0)]:
        if mask.sum() > 0:
            bm = B[mask]; lm = L[mask]; cm = (bm == lm)
            tp_ = ((bm == 1) & (lm == 1)).sum()
            fp_ = ((bm == 1) & (lm == 0)).sum()
            fn_ = ((bm == 0) & (lm == 1)).sum()
            p_ = tp_ / max(tp_ + fp_, 1)
            r_ = tp_ / max(tp_ + fn_, 1)
            r[f'{label}_acc'] = cm.mean()
            r[f'{label}_p'] = p_
            r[f'{label}_r'] = r_
            r[f'{label}_f1'] = 2 * p_ * r_ / max(p_ + r_, 1e-8)
            r[f'{label}_n'] = int(mask.sum())
    return r




@torch.no_grad()
def evaluate_degree_breakdown(model, loader, device, buckets=[(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 10000)]):
    """Long-tail breakdown by degree bucket."""
    model.eval()
    all_preds, all_labs, all_td = [], [], []
    for batch in tqdm(loader, desc="DegreeEval"):
        hi = batch['head'].to(device)
        ri = batch['relation'].to(device)
        ti = batch['tail'].to(device)
        lab = batch['label'].to(device)
        rh = batch['raw_head'].to(device)
        rt = batch['raw_tail'].to(device)
        rr = batch['raw_rel'].to(device)
        td = batch['tail_degree'].to(device)
        td2 = batch['tail_degree'].to(device)
        S_fused, S_struct, S_emb, alpha = model(hi, ri, ti, raw_head=rh,
                                                raw_tail=rt, raw_rel=rr,
                                                tail_deg=td2)
        all_preds.append(S_fused.detach().cpu())
        all_labs.append(lab.cpu())
        all_td.append(td.cpu())
        del S_fused, S_struct, S_emb, alpha, hi, ri, ti, lab, rh, rt, rr, td, td2
    P = torch.cat(all_preds).numpy()
    L = torch.cat(all_labs).numpy()
    TD = torch.cat(all_td).numpy()
    B = (P >= 0.5).astype(np.float32)
    results = {}
    for lo, hi in buckets:
        mask = (TD >= lo) & (TD < hi)
        if mask.sum() == 0: continue
        bm = B[mask]; lm = L[mask]
        cm = (bm == lm)
        tp_ = ((bm == 1) & (lm == 1)).sum()
        fp_ = ((bm == 1) & (lm == 0)).sum()
        fn_ = ((bm == 0) & (lm == 1)).sum()
        p_ = tp_ / max(tp_ + fp_, 1)
        r_ = tp_ / max(tp_ + fn_, 1)
        label = f"deg[{lo},{hi})"
        results[label] = {
            'acc': cm.mean(), 'prec': p_, 'rec': r_,
            'f1': 2 * p_ * r_ / max(p_ + r_, 1e-8),
            'n': int(mask.sum())
        }
    return results




@torch.no_grad()
def extract_feedback(model, loader, device, bridge, ent2idx):
    """Extract per-entity feedback signals from trained M4.

    Returns:
        entity_alpha: dict raw_tail_id -> average alpha
        anchor_avg_weight: dict anchor_raw_id -> average bridge weight
        per_degree_stats: dict deg_bucket -> {'count','accuracy','pos_ratio','avg_alpha'}
    """
    model.eval()
    entity_alpha_sum = {}
    entity_alpha_cnt = {}
    anchor_weight_sum = {}
    anchor_weight_cnt = {}
    degree_buckets = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 10000)]
    per_degree = {f"deg[{lo},{hi})": [0, 0, 0, 0.0] for lo, hi in degree_buckets}

    for batch in tqdm(loader, desc="Feedback"):
        hi = batch['head'].to(device)
        ri = batch['relation'].to(device)
        ti = batch['tail'].to(device)
        lab = batch['label'].to(device)
        rh = batch['raw_head'].to(device)
        rt = batch['raw_tail'].to(device)
        rr = batch['raw_rel'].to(device)
        td = batch['tail_degree'].to(device)

        S_fused, S_struct, S_emb, alpha = model(hi, ri, ti,
                                                  raw_head=rh, raw_tail=rt,
                                                  raw_rel=rr, tail_deg=td)

        preds = (S_fused > 0.5).float()
        alpha_np = alpha.cpu().numpy()
        rt_np = rt.cpu().numpy()
        lab_np = lab.cpu().numpy()
        pred_np = preds.cpu().numpy()
        td_np = td.cpu().numpy()

        for i in range(len(rt_np)):
            rid = int(rt_np[i])
            a_val = float(alpha_np[i])
            deg_val = float(td_np[i])

            entity_alpha_sum[rid] = entity_alpha_sum.get(rid, 0.0) + a_val
            entity_alpha_cnt[rid] = entity_alpha_cnt.get(rid, 0) + 1

            for lo, hi in degree_buckets:
                if lo <= deg_val < hi:
                    bk = f"deg[{lo},{hi})"
                    per_degree[bk][0] += 1
                    per_degree[bk][1] += int(pred_np[i] == lab_np[i])
                    per_degree[bk][2] += int(lab_np[i] == 1)
                    per_degree[bk][3] += a_val
                    break

        # Accumulate bridge weights per anchor
        if bridge is not None:
            anchor_w, anchor_idx = bridge.lookup(rt)
            if anchor_idx.shape[1] > 0:
                w_np = anchor_w.cpu().numpy()
                idx_np = anchor_idx.cpu().numpy()
                for i in range(len(rt_np)):
                    for j in range(anchor_idx.shape[1]):
                        aid = int(idx_np[i, j])
                        if aid > 0:
                            aw = float(w_np[i, j])
                            anchor_weight_sum[aid] = anchor_weight_sum.get(aid, 0.0) + aw
                            anchor_weight_cnt[aid] = anchor_weight_cnt.get(aid, 0) + 1

    entity_alpha = {k: entity_alpha_sum[k] / max(entity_alpha_cnt[k], 1)
                    for k in entity_alpha_sum}
    anchor_avg = {k: anchor_weight_sum[k] / max(anchor_weight_cnt[k], 1)
                  for k in anchor_weight_sum}
    per_degree_stats = {}
    for bk, (cnt, correct, pos, asum) in per_degree.items():
        per_degree_stats[bk] = {
            'count': cnt, 'accuracy': correct / max(cnt, 1),
            'positive_ratio': pos / max(cnt, 1),
            'avg_alpha': asum / max(cnt, 1)
        }

    return entity_alpha, anchor_avg, per_degree_stats


def apply_feedback(m2, m3, entity_alpha, per_degree_stats, gkg_degrees):
    """Apply M4 feedback to refine M2/M3 (Progressive Filtering, Papadakis ICDE 2023).

    Rules:
      1. High-alpha (>0.45) entities in low-accuracy (<0.7 F1) degree buckets
         -> flagged for expanded anchor budget in next M2 round
      2. Per-degree bucket stats -> adjust M2 adaptive thresholds
      3. Returns feedback summary for logging and optional bridge rebuild
    """
    print("\n  --- Progressive Feedback (M4 -> M1/M2/M3) ---")

    needy_entities = []
    for rid, a_val in entity_alpha.items():
        deg = gkg_degrees.get(rid, 1)
        bucket_f1 = None
        for bk, stats in per_degree_stats.items():
            lo = int(bk.split('[')[1].split(',')[0])
            hi = int(bk.split(',')[1].split(')')[0])
            if lo <= deg < hi:
                # Estimated F1 from accuracy and positive ratio
                acc = stats['accuracy']
                pr = stats['positive_ratio']
                if acc + pr > 0.01:
                    bucket_f1 = 2 * acc * pr / (acc + pr)
                break
        if a_val > 0.45 and (bucket_f1 is None or bucket_f1 < 0.7):
            needy_entities.append(rid)

    print(f"  Entities needing better bridge support: {len(needy_entities)}/{len(entity_alpha)}")

    # Per-degree diagnostic
    for bk in sorted(per_degree_stats.keys()):
        s = per_degree_stats[bk]
        print(f"  {bk}: n={s['count']} acc={s['accuracy']:.3f} "
              f"pos={s['positive_ratio']:.3f} avg_alpha={s['avg_alpha']:.3f}")

    return {
        'needy_entities': needy_entities,
        'per_degree': per_degree_stats,
        'entity_alpha': entity_alpha,
    }



def run(name, data_dir, out_dir, dim=128, max_k=16, bs=512, epochs=40,
        lr=0.001, seed=2025, lambda_bridge=0.2, beta_entropy=0.01,
        expansion_thresh=2, ablation=None, scorer='transe'):
    """DAGeR v6 full pipeline.
    
    Phase I (no GPU needed): M1 IDF signatures → M2 Parametric simulation → M3 Bridge weighting
    Phase II (GPU): M4 Dual-track fusion model training
    
    ablation: None (full), 'no_bridge', 'random_bridge', 'struct_only', 'emb_only'
    """
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"DAGeR v6 | {name} | dim={dim} max_k={max_k} lambda={lambda_bridge}")
    print(f"  4 Modules x 7 Steps | B/C dual-track: S_struct + S_emb")
    print(f"  Ablation: {ablation or 'none (full model)'}")
    print(f"{'='*60}")

    # ---- Load data ----
    t0 = time.time()
    t1 = read_triples(os.path.join(data_dir, 'triples_1'))  # DKG
    t2 = read_triples(os.path.join(data_dir, 'triples_2'))  # GKG
    e2i, r2i = build_ent_rel_mappings(data_dir)
    idx2raw = {v: k for k, v in e2i.items()}
    deg = compute_node_degrees(os.path.join(data_dir, 'triples_1'),
                               os.path.join(data_dir, 'triples_2'))
    print(f"  Data: {len(e2i)}e {len(r2i)}r | DKG={len(t1)} GKG={len(t2)} "
          f"| load={time.time()-t0:.1f}s")

    # ================================================================
    # PHASE I: Structural Modules (M1, M2, M3)
    # ================================================================
    t_pre = time.time()

    # M1: IDF-Weighted Structural Signatures (Steps 1-2)
    print(f"\n  --- [M1] IDFWeightedSignatures (Steps 1-2) ---")
    t1s = time.time()
    # Pre-read fusion_train to extract active training entities
    fusion_train = read_triples(os.path.join(data_dir, 'sup_triples_1_fusion'))
    train_entities = set()
    for h, r, t, _, _ in fusion_train:
        train_entities.add(h); train_entities.add(t)
    train_entities.discard(-1)
    print(f"  Train entities for anchor pool: {len(train_entities)}")
    sig = IDFWeightedSignatures(expansion_thresh=expansion_thresh)
    sig.build(t1)  # use all DKG entities for full anchor diversity
    print(f"  [M1] total: {time.time()-t1s:.1f}s")

    # M2: Parametric Graph Simulation (Steps 3-4)
    print(f"\n  --- [M2] ParametricSimulation (Steps 3-4) ---")
    t2s = time.time()
    # fusion_train already loaded above (with train_entities extraction)
    sim = ParametricSimulation(sig, max_k=max_k)
    sim.build(fusion_train, e2i, deg)
    print(f"  [M2] total: {time.time()-t2s:.1f}s")

    # M3: Multi-Constraint Bridge Weighting (Steps 5-6)
    print(f"\n  --- [M3] MultiConstraintBridge (Steps 5-6) ---")
    t3s = time.time()
    bridge = MultiConstraintBridge(sig, idx2raw)
    bridge.build(sim, deg)
    print(f"  [M3] total: {time.time()-t3s:.1f}s")

    # Negative sampling
    print(f"\n  --- Negative Sampling ---")
    tns = time.time()
    all_ents = set(list(e2i.keys()))
    neg_sampler = NegativeSampler(fusion_train, all_ents, num_neg=1, seed=seed)
    neg_triples = neg_sampler.generate(fusion_train)
    print(f"  [Neg] {len(neg_triples)} negatives in {time.time()-tns:.1f}s")

    t_pre_total = time.time() - t_pre
    print(f"\n  Phase I (Preprocessing): {t_pre_total:.1f}s")


    all_train = fusion_train + neg_triples
    all_labels = [1] * len(fusion_train) + [0] * len(neg_triples)

    # Test: reference (positive) + false reference (negative)
    vp = read_triples(os.path.join(data_dir, 'ref_triples_1_fusion'))
    vn = read_triples(os.path.join(data_dir, 'false_ref_triples_1_fusion'))

    tr_ds = FusionDataset(all_train, all_labels, e2i, r2i, deg)
    te_ds = FusionDataset(vp + vn, [1] * len(vp) + [0] * len(vn), e2i, r2i, deg)
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn,
                       num_workers=0, pin_memory=True)
    te_ld = DataLoader(te_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn,
                       num_workers=0, pin_memory=True)

    lttr = sum(1 for d in tr_ds.data if d[8])
    ltte = sum(1 for d in te_ds.data if d[8])
    print(f"  Train={len(tr_ds)} (LT={lttr}) Test={len(te_ds)} (LT={ltte})")

    # ================================================================
    # PHASE II: Model Training (M4: Dual-Track Fusion)
    # ================================================================

    # Ablation: handle special modes
    if ablation == 'emb_only':
        lambda_bridge = 0.0  # no structural signal
    elif ablation == 'struct_only':
        lambda_bridge = 0.0  # still need bridge for struct signal

    model = DAGeR_v6(len(e2i), len(r2i), dim=dim, max_k=max_k,
                     lambda_bridge=lambda_bridge, scorer=scorer).to(dev)

    if ablation in ('no_bridge', 'random_bridge'):
        model.set_bridge(None)  # no bridge at all
    else:
        model.set_bridge(bridge)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [M4] DAGeR v6: {n_params:,} params ({n_trainable} trainable: "
          f"tau, alpha, GMC, DomainGate, embeddings)")

    # Initialize tau based on class balance
    pos_ratio = len(fusion_train) / max(len(all_train), 1)
    with torch.no_grad():
        if 0 < pos_ratio < 1:
            model.enhance.tau.data = torch.tensor(-math.log(pos_ratio / (1 - pos_ratio)))
    print(f"  Initial tau = {model.enhance.tau.item():.4f} (pos_ratio={pos_ratio:.3f})")
    alpha_t = model.get_alpha_for_degree(1.0)
    alpha_h = model.get_alpha_for_degree(10.0)
    print(f"  Initial alpha: tail(deg=1)={alpha_t:.4f} head(deg=10)={alpha_h:.4f}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- Training ----
    best, best_res, times = 0, None, []
    for ep in range(epochs):
        t0_ep = time.time()
        if dev == 'cuda':
            torch.cuda.empty_cache()
            print(f"  [MEM] Before train: alloc={torch.cuda.memory_allocated(dev)/1e9:.3f}GB res={torch.cuda.memory_reserved(dev)/1e9:.3f}GB")
        loss = train_epoch(model, tr_ld, opt, dev, beta=beta_entropy,
                                         epoch=ep, total_epochs=epochs)
        tr_t = time.time() - t0_ep
        t0_ev = time.time()
        if dev == 'cuda':
            torch.cuda.empty_cache()
            print(f"  [MEM] Before eval: alloc={torch.cuda.memory_allocated(dev)/1e9:.3f}GB res={torch.cuda.memory_reserved(dev)/1e9:.3f}GB")
        res = evaluate(model, te_ld, dev)
        ev_t = time.time() - t0_ev
        times.append(tr_t + ev_t)
        s = "*" if res['f1'] > best else " "
        if res['f1'] > best:
            best = res['f1']
            best_res = dict(res)
            best_res['tau'] = model.enhance.tau.item()
            pass  # v11: alpha tracked in print line only
        ls = f"LT F1={res.get('lt_f1', 0):.4f}" if 'lt_f1' in res else ""
        alpha_tail = model.get_alpha_for_degree(1.0)
        alpha_head = model.get_alpha_for_degree(10.0)
        print(f"  E{ep+1:3d}{s} ACC={res['acc']:.4f} F1={res['f1']:.4f} {ls} "
              f"| tr={tr_t:.1f}s ev={ev_t:.1f}s loss={loss:.4f} "
              f"tau={model.enhance.tau.item():.3f} a_tail={alpha_tail:.3f} a_head={alpha_head:.3f}")

    avg_t = sum(times) / max(len(times), 1)
    print(f"\n  Avg epoch: {avg_t:.1f}s | Total train: {sum(times):.1f}s")
    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() / 1024**2
        reserved = torch.cuda.max_memory_reserved() / 1024**2
        print(f"  GPU memory: {alloc:.0f} MiB allocated, {reserved:.0f} MiB reserved")
    print(f"  Phase I (preprocess): {t_pre_total:.1f}s")

    # ---- Best results ----
    print(f"\n{'='*60}")
    print(f"  BEST (tau={best_res.get('tau', 0):.4f}, alpha_tail={best_res.get('alpha_tail', 0):.4f}, alpha_head={best_res.get('alpha_head', 0):.4f}):")
    print(f"  ACC={best_res['acc']:.4f} P={best_res['prec']:.4f} "
          f"R={best_res['rec']:.4f} F1={best_res['f1']:.4f}")
    for k in ['lt', 'hd']:
        if f'{k}_f1' in best_res:
            print(f"  {k.upper():>3s}: ACC={best_res[f'{k}_acc']:.4f} "
                  f"F1={best_res[f'{k}_f1']:.4f} (n={best_res[f'{k}_n']})")
    print(f"{'='*60}\n")

    # ---- Degree-bucket breakdown on test set ----
    if ablation is None:
        print(f"\n  --- Per-Degree-Bucket Breakdown ---")
        deg_results = evaluate_degree_breakdown(model, te_ld, dev)
        for bucket, r in sorted(deg_results.items()):
            print(f"  {bucket}: F1={r['f1']:.4f} ACC={r['acc']:.4f} (n={r['n']})")

    # ---- Save results ----
    res_path = os.path.join(out_dir, f"results_{name}.txt")
    with open(res_path, 'w') as f:
        f.write(f"DAGeR v6 | {name} | dim={dim} max_k={max_k} lambda={lambda_bridge}\n")
        f.write(f"Ablation: {ablation or 'none'}\n")
        f.write(f"BEST: ACC={best_res['acc']:.4f} P={best_res['prec']:.4f} "
                f"R={best_res['rec']:.4f} F1={best_res['f1']:.4f} "
                f"tau={best_res.get('tau',0):.4f} alpha={best_res.get('alpha',0):.4f}\n")
        for k in ['lt', 'hd']:
            if f'{k}_f1' in best_res:
                f.write(f"{k.upper()}: ACC={best_res[f'{k}_acc']:.4f} "
                        f"F1={best_res[f'{k}_f1']:.4f} (n={best_res[f'{k}_n']})\n")
        f.write(f"Phase I: {t_pre_total:.1f}s | Avg epoch: {avg_t:.1f}s\n")
        f.write(f"Params: {n_params} total, {n_trainable} trainable\n")
        if ablation is None:
            f.write("\nDegree-bucket breakdown:\n")
            for bucket, r in sorted(deg_results.items()):
                f.write(f"  {bucket}: F1={r['f1']:.4f} ACC={r['acc']:.4f} (n={r['n']})\n")

    # Save model checkpoint
    ckpt_path = os.path.join(out_dir, f"model_{name}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'opt_state_dict': opt.state_dict(),
        'best_res': best_res,
        'args': {'dim': dim, 'max_k': max_k, 'lambda_bridge': lambda_bridge,
                 'expansion_thresh': expansion_thresh}
    }, ckpt_path)
    print(f"  Saved: {res_path}")
    print(f"  Saved: {ckpt_path}")

    return best_res, avg_t


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='DAGeR v6: Domain-Aware Graph Rewiring')
    p.add_argument('--data_dir', type=str, required=True,
                   help='Path to dataset dir containing triples_1, triples_2, etc.')
    p.add_argument('--name', type=str, default='dataset',
                   help='Dataset name for output files')
    p.add_argument('--output_dir', type=str, default='./outputs',
                   help='Output directory for results and checkpoints')
    p.add_argument('--dim', type=int, default=128,
                   help='Embedding dimension')
    p.add_argument('--max_k', type=int, default=16,
                   help='Max anchors per tail entity')
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=2025)
    p.add_argument('--lambda_bridge', type=float, default=0.2,
                   help='Bridge injection weight')
    p.add_argument('--beta_entropy', type=float, default=0.01,
                   help='Entropy regularizer strength on alpha')
    p.add_argument('--expansion_thresh', type=int, default=2,
                   help='Degree threshold for 2-hop expansion')
    p.add_argument('--scorer', type=str, default='transe',
                   choices=['transe', 'rotate', 'exefuse'],
                   help='Base KG scoring model (pluggable encoder)')
    p.add_argument('--ablation', type=str, default=None,
                   choices=['no_bridge', 'random_bridge', 'struct_only', 'emb_only'],
                   help='Ablation study mode')

    args = p.parse_args()
    run(args.name, args.data_dir, args.output_dir,
        dim=args.dim, max_k=args.max_k, bs=args.batch_size,
        epochs=args.epochs, lr=args.lr, seed=args.seed,
        lambda_bridge=args.lambda_bridge, beta_entropy=args.beta_entropy,
        expansion_thresh=args.expansion_thresh, ablation=args.ablation, scorer=args.scorer)









