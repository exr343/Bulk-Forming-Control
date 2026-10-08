"""Coil GNN with TEMPERATURE IN THE NODAL STATE (decided 2026-10-07): the same
three-stage pipeline as coil.py (inputs, die push, stages, split, sweep), except

  state per surface node = displacement (3) + temperature (1)
  hit step     predicts the change of both (4 outputs, output-normalized; in the
               loss temperature counts like one displacement component)
  reheat step  temperature after = hotter of the temperature before and the coil
               profile, exactly the simulator's rule (checked: 0.000 C); the GNN
               predicts only the reheat's shape change (its temperature output
               is unused there)
  temperature input = the real surface temperature at the start of the step
               (in practice an IR camera at each scan; the bookkept "coil
               temperature" of coil.py is gone); old-simulator data reheated to
               its starting profile before every hit, so its input is that
               profile and its target the saved temperature after the hit
  cycle chains feed the predicted temperature forward with the predicted shape.

Per-node inputs (16) as in coil.py, with the temperature column = current temperature.
Written to GNN/coil_T_sweep* (coil.py and its sweeps are left untouched).

Usage (from the repo root):
    python -m applications.Agility_Forge.GNN.coil_T --stage 12 --mp 5 --test-run square --out <dir>
    python -m applications.Agility_Forge.GNN.coil_T --stage 3 --mp 5 --test-run square --out <dir>
    python -m applications.Agility_Forge.GNN.coil_T --summarize --out <dir>
"""

import argparse
import dataclasses
import glob
import json
import logging
import os
import random
import shutil
import time
from typing import Optional

import meshio
import numpy as np
import torch
import torch.nn as nn

from applications.Agility_Forge.GNN.data import build_surface_mesh_info
from applications.Agility_Forge.GNN.model import EncodeProcessDecode
from applications.Agility_Forge.GNN.normalizer import OnlineNormalizer
from applications.Agility_Forge.GNN.train import _chamfer_hausdorff_mm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("coil")

AF = "applications/Agility_Forge"
PRE_DIR = f"{AF}/data/dataset_pretraining"
PRE_ALLOW = f"{AF}/GNN/mp_sweep/rollout_snapshot_383.json"
OLD_SQ = f"{AF}/data/dataset_finetuning/finished"
NEW_SQ = f"{AF}/data/dataset_die12_coil_5pass"   # 5-pass runs (pass 5 half-gap 5.0-5.3 mm), keep-the-hotter reheat
OUT = f"{AF}/GNN/coil_T_sweep_test_square"
TRAIN_SEEDS, VAL_SEED, TEST_SEED = [None, 1, 2, 4, 5, 6, 7, 8], 9, 3   # set_split() changes the test run


def set_split(test):
    """Test run = jitter seed `test` (int) or the plain square (None); it is held
    out of stages 2 and 3, validation stays seed 9, every other run trains."""
    global TRAIN_SEEDS, TEST_SEED
    TEST_SEED = test
    TRAIN_SEEDS = [s for s in [None, 1, 2, 3, 4, 5, 6, 7, 8] if s != test]


def run_name(seed):
    return "square" if seed is None else f"seed{seed}"

DIE_W = 12.7
SOFT_MM = 0.25
U_MIN, U_RANGE = 0.5, 1.5
DT_SCALE = 300.0
N_RAW_FLAGS, N_NORMED, N_RAW_TAIL = 4, 8, 4
NODE_IN, EDGE_IN, OUT_DIM = N_RAW_FLAGS + N_NORMED + N_RAW_TAIL, 8, 4     # outputs: dx, dy, dz, dT


def run_dir(root, seed):
    return f"{root}/square" if seed is None else f"{root}/square_jitter_seed{seed}"


# ----------------------------------------------------------------------------
# Hits, features
# ----------------------------------------------------------------------------
@dataclasses.dataclass
class Hit:
    """One die action. `center` (current x, new die) or `band` (rest x, old)."""
    angle_deg: float
    stroke_mm: float
    center: Optional[float] = None
    band: Optional[tuple] = None


class Mesh:
    """Surface graph + static flags, as torch tensors on `device`."""

    def __init__(self, reference_vtu, device):
        mi = build_surface_mesh_info(reference_vtu)
        self.info = mi
        self.rest = mi.rest_pos.to(device)
        self.flags = torch.cat([mi.is_bottom_face, mi.is_pin_node, mi.is_lateral_wall], dim=-1).to(device)
        self.wall = mi.is_lateral_wall[:, 0].to(device) > 0
        self.senders, self.receivers = mi.senders.to(device), mi.receivers.to(device)
        self.full_to_surface = mi.full_to_surface
        self.n = self.rest.shape[0]


def node_features(mesh, x_k, T, hit: Optional[Hit], dT=None):
    """Raw (N, 16) inputs for one step. `hit=None` = reheat step (needs dT).
    Differentiable in x_k and in a hit's angle/stroke/centre tensors."""
    dev = x_k.device
    n = mesh.n
    if hit is None:
        flag = torch.ones(n, 1, device=dev)
        push = torch.zeros(n, 3, device=dev)
        dist = torch.zeros(n, 1, device=dev)
        tail = torch.cat([dT[:, None] / DT_SCALE, torch.zeros(n, 3, device=dev)], dim=-1)
    else:
        flag = torch.zeros(n, 1, device=dev)
        pos = mesh.rest + x_k
        if hit.center is not None:
            xs = pos[:, 0]
            c = torch.as_tensor(hit.center, dtype=torch.float32, device=dev)
            lo, hi = c - 0.5 * DIE_W, c + 0.5 * DIE_W
        else:
            xs = mesh.rest[:, 0]
            lo = torch.as_tensor(hit.band[0], dtype=torch.float32, device=dev)
            hi = torch.as_tensor(hit.band[1], dtype=torch.float32, device=dev)
        a = torch.deg2rad(torch.as_tensor(hit.angle_deg, dtype=torch.float32, device=dev))
        u = torch.as_tensor(hit.stroke_mm, dtype=torch.float32, device=dev)
        y = torch.cos(a) * pos[:, 1] - torch.sin(a) * pos[:, 2]       # scipy from_euler("x", a) frame
        hard = mesh.wall & (xs >= lo) & (xs <= hi)
        if not bool(hard.any()):
            # No side-wall node under the die: never in the data (>= 6 nodes), but a
            # predicted shape in a cycle chain can drift there. Use the nearest
            # side-wall nodes; their soft weight below makes the push ~0.
            d = (xs - 0.5 * (lo + hi)).abs().detach()
            hard = mesh.wall & (d <= d[mesh.wall].min() + 1.0)
        soft = mesh.wall.float() * torch.sigmoid((xs - lo) / SOFT_MM) * torch.sigmoid((hi - xs) / SOFT_MM)
        y_lo, y_hi = y[hard].min(), y[hard].max()
        mag = (torch.relu(y_lo + u - y) - torch.relu(y - (y_hi - u))) * soft
        e = torch.stack([torch.zeros_like(a), torch.cos(a), -torch.sin(a)])
        push = mag[:, None] * e[None, :]
        dist = (xs - 0.5 * (lo + hi))[:, None]
        tail = torch.cat([torch.zeros(n, 1, device=dev),
                          torch.stack([torch.sin(a), torch.cos(a), (u - U_MIN) / U_RANGE]).expand(n, 3)], dim=-1)
    return torch.cat([mesh.flags, flag, push, dist, x_k, T[:, None], tail], dim=-1)


class CoilGNN(nn.Module):
    def __init__(self, mesh: Mesh, mp_steps, latent=128, layers=2):
        super().__init__()
        self.register_buffer("rest", mesh.rest.clone())
        self.register_buffer("senders", mesh.senders.clone())
        self.register_buffer("receivers", mesh.receivers.clone())
        self.node_norm = OnlineNormalizer(N_NORMED)
        self.edge_norm = OnlineNormalizer(EDGE_IN)
        self.out_norm = OnlineNormalizer(OUT_DIM)
        self.core = EncodeProcessDecode(NODE_IN, EDGE_IN, OUT_DIM, latent_size=latent, num_layers=layers,
                                        message_passing_steps=mp_steps)
        self.mp_steps = mp_steps

    def freeze_normalizers(self):
        for nrm in (self.node_norm, self.edge_norm, self.out_norm):
            nrm.max_accumulations = int(nrm.num_accumulations.item())

    def _edges(self, x_k):
        s, r = self.senders, self.receivers
        rel_m = (self.rest[s] - self.rest[r]).unsqueeze(0).expand(x_k.shape[0], -1, -1)
        w = self.rest.unsqueeze(0) + x_k
        rel_w = w[:, s] - w[:, r]
        raw = torch.cat([rel_m, rel_m.norm(dim=-1, keepdim=True), rel_w, rel_w.norm(dim=-1, keepdim=True)], dim=-1)
        return self.edge_norm(raw, accumulate=self.training)

    def delta_normed(self, feats, x_k):
        """feats (B, N, 16), x_k (B, N, 3) -> predicted change, output-normalized."""
        a, b = N_RAW_FLAGS, N_RAW_FLAGS + N_NORMED
        nf = torch.cat([feats[..., :a], self.node_norm(feats[..., a:b], accumulate=self.training), feats[..., b:]], -1)
        return self.core(nf, self._edges(x_k), self.senders, self.receivers)

    def step(self, feats, x_k, T_k):
        """Hit step: physical next displacement and temperature."""
        d = self.out_norm.inverse(self.delta_normed(feats, x_k))
        return x_k + d[..., :3], T_k + d[..., 3]

    def step_shape(self, feats, x_k):
        """Reheat step: next displacement only (the temperature follows the reheat rule)."""
        return x_k + self.out_norm.inverse(self.delta_normed(feats, x_k))[..., :3]


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
class VtuCache:
    def __init__(self, mesh: Mesh):
        self.idx = mesh.full_to_surface
        self.rest = mesh.rest.detach().cpu()
        self.d = {}

    def get(self, path):
        if path not in self.d:
            m = meshio.read(path)
            self.d[path] = (torch.tensor(m.point_data["Displacement"][self.idx], dtype=torch.float32),
                            torch.tensor(m.point_data["Temperature"][self.idx, 0], dtype=torch.float32))
        return self.d[path]


@dataclasses.dataclass
class Step:
    """One transition: input state `x_in` -> `x_out` (vtu paths). `T_in` = file
    whose temperature is the step's temperature input: for a hit, the
    temperature at its start (x_in; the starting profile for old data); for a
    reheat (hit None), the temperature after it (x_out, the rule's result),
    with `T_old` = the temperature before it (x_in)."""
    x_in: str
    x_out: str
    T_in: str
    hit: Optional[Hit]
    T_old: Optional[str] = None
    run: str = ""
    label: str = ""


def old_steps(dataset_dir, rollouts=None):
    """Old-simulator hits (band die, reheat to the starting profile before
    every hit): one Step per hit, grouped by rollout."""
    man = json.load(open(f"{dataset_dir}/manifest.json"))
    by = {}
    for r in man["records"]:
        by.setdefault(r["rollout"], {})[r["hit"]] = r
    out = {}
    for rid, hits in sorted(by.items()):
        if rollouts is not None and rid not in rollouts:
            continue
        if 0 not in hits:
            continue
        und = os.path.join(dataset_dir, hits[0]["vtu_path"])
        steps = []
        for h in range(1, max(hits) + 1):
            if h not in hits or h - 1 not in hits:
                break
            r = hits[h]
            steps.append(Step(os.path.join(dataset_dir, hits[h - 1]["vtu_path"]), os.path.join(dataset_dir, r["vtu_path"]),
                              und, Hit(r["R_j_deg"], r["u_j_mm"], band=(r["d_j_mm"], r["x_max_band_mm"])),
                              run=f"{dataset_dir}#{rid}", label=f"hit {h}"))
        out[rid] = steps
    return out


def new_cycles(run, cache):
    """New-simulator run -> list of cycles; each cycle = list of Steps, the
    reheat step first (none for the first cycle), and a flag for whether the
    cycle is complete (all its planned hits were run). Temperatures are the
    simulator's real ones (what an IR camera would read)."""
    man = json.load(open(f"{run}/manifest.json"))
    recs = {r["hit"]: r for r in man["records"] if r["kind"] == "hit_final"}
    reh = {}
    for r in man["reheats"]:                 # several reheats before the same hit: keep the last
        reh[r["hit"]] = r
    starts = sorted(h for h in reh if h in recs)
    P = lambda rel: os.path.join(run, rel)
    und = P(recs[0]["vtu_path"]) if 0 in recs else P("rollout_01/undeformed.vtu")
    sched = json.loads(str(np.load(f"{run}/rollout_01/checkpoint.npz")["sched"]))
    cycles = []
    for i, h0 in enumerate(starts):
        h1 = (starts[i + 1] if i + 1 < len(starts) else max(recs) + 1)
        steps = []
        if h0 > 1:
            pre, post = P(recs[h0 - 1]["vtu_path"]), P(reh[h0]["vtu_path"])
            steps.append(Step(pre, post, post, None, T_old=pre, run=run, label=f"reheat before hit {h0}"))
        for h in range(h0, h1):
            if h not in recs:
                break
            r = recs[h]
            x_in = (und if h == 1 else P(reh[h]["vtu_path"]) if h == h0 else P(recs[h - 1]["vtu_path"]))
            steps.append(Step(x_in, P(r["vtu_path"]), x_in, Hit(r["R_j_deg"], r["u_j_mm"], center=r["die_center_mm"]),
                              run=run, label=f"hit {h}"))
        # Complete = every planned hit of the cycle was run: a later cycle
        # started, or the run went on to plan another cycle (e.g. it stopped on
        # that cycle's reheat), or the current plan is used up.
        last = i + 1 == len(starts)
        cyc_no = recs[h1 - 1].get("cycle")
        complete = (not last) or sched["cycle"] > cyc_no or sched["pos"] >= len(sched["plan"])
        cycles.append({"steps": steps, "complete": bool(complete), "start_hit": h0})
    return cycles


def features_for(mesh, cache, step: Step, x_k, dev, T_k=None):
    """Inputs for one step; `T_k` = current (e.g. predicted) temperature for a
    hit, else the recorded one. A reheat uses the recorded before/after
    temperatures (it always starts a chain from the true state)."""
    if step.hit is None:
        T_new = cache.get(step.T_in)[1].to(dev)
        return node_features(mesh, x_k, T_new, None, dT=T_new - cache.get(step.T_old)[1].to(dev))
    T = cache.get(step.T_in)[1].to(dev) if T_k is None else T_k
    return node_features(mesh, x_k, T, step.hit)


def batch_tensors(mesh, cache, steps, dev):
    """One-hit examples (hit steps only): features, start displacement, and
    targets (change of displacement, change of temperature)."""
    feats, xs, tgts = [], [], []
    for s in steps:
        x0 = cache.get(s.x_in)[0].to(dev)
        x1, T1 = (a.to(dev) for a in cache.get(s.x_out))
        T0 = cache.get(s.T_in)[1].to(dev)
        feats.append(features_for(mesh, cache, s, x0, dev)); xs.append(x0)
        tgts.append(torch.cat([x1 - x0, (T1 - T0)[:, None]], dim=-1))
    return torch.stack(feats), torch.stack(xs), torch.stack(tgts)


def onehit_loss(model, mesh, cache, steps, dev):
    f, x, d = batch_tensors(mesh, cache, steps, dev)
    pred = model.delta_normed(f, x)
    return ((pred - model.out_norm(d, accumulate=model.training)) ** 2).sum(-1).mean()


def chain(model, mesh, cache, steps, dev, skip_reheat=False):
    """Roll a cycle from its true starting state (shape + measured temperature),
    each step fed the previous prediction of both. A reheat's temperature is
    the rule's (= recorded) value. Returns lists of predicted and true
    (displacement, temperature) after each step."""
    x, T = (a.to(dev) for a in cache.get(steps[0].x_in))
    preds, trues = [], []
    for s in steps:
        if s.hit is None:
            T_next = cache.get(s.x_out)[1].to(dev)
            x_next = x if skip_reheat else model.step_shape(features_for(mesh, cache, s, x, dev)[None], x[None])[0]
        else:
            xn, Tn = model.step(features_for(mesh, cache, s, x, dev, T_k=T)[None], x[None], T[None])
            x_next, T_next = xn[0], Tn[0]
        preds.append((x_next, T_next)); trues.append(tuple(a.to(dev) for a in cache.get(s.x_out)))
        x, T = x_next, T_next
    return preds, trues


def chain_loss(model, mesh, cache, steps, dev):
    """Mean over steps of the per-node squared error of displacement and
    temperature, each in output-normalized units."""
    preds, trues = chain(model, mesh, cache, steps, dev)
    std = model.out_norm._std()
    return torch.stack([((((p - t) / std[:3]) ** 2).sum(-1) + ((pT - tT) / std[3]) ** 2).mean()
                        for (p, pT), (t, tT) in zip(preds, trues)]).mean()


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_loop(model, train_items_fn, val_fn, lr, max_epochs, patience, out_dir, tag):
    """train_items_fn(epoch) -> list of zero-arg callables returning a loss
    (one optimizer step each); val_fn() -> float."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.eval()
    with torch.no_grad():
        best = val_fn() if tag != "stage1" else float("inf")
    best_state, best_epoch, since, hist = None, 0, 0, []
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        model.train()
        items = train_items_fn(ep)
        tot = 0.0
        for item in items:
            loss = item()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            v = val_fn()
        hist.append({"epoch": ep, "train_loss": tot / max(len(items), 1), "val_loss": v})
        log.info(f"[{tag}] epoch {ep:3d} train {tot / max(len(items), 1):.4e} val {v:.4e}")
        if v < best:
            best, best_epoch, since = v, ep, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    info = {"best_epoch": best_epoch, "best_val_loss": best, "epochs_run": len(hist),
            "elapsed_s": time.time() - t0, "history": hist}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(info, open(f"{out_dir}/{tag}_training.json", "w"), indent=1)
    return info


def batches(steps, size, rng):
    steps = list(steps)
    rng.shuffle(steps)
    return [steps[i:i + size] for i in range(0, len(steps), size)]


def load_old_random():
    allow = set(json.load(open(PRE_ALLOW)))
    by = old_steps(PRE_DIR, allow)
    ids = sorted(by)
    n_tr = max(1, int(round(0.8 * len(ids))))                  # train.py's prefix split
    train = [s for i in ids[:n_tr] for s in by[i]]
    test = [s for i in ids[n_tr:] for s in by[i]]
    return train, test


def save(model, path, extra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mp_steps": model.mp_steps, **extra}, path)


def load(mesh, path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = CoilGNN(mesh, ck["mp_steps"]).to(dev)
    m.load_state_dict(ck["state_dict"])
    return m


def stage12(args, mesh, cache, dev):
    out = f"{args.out}/mp_{args.mp}"
    rng = random.Random(args.seed)
    rnd_train, rnd_test = load_old_random()
    log.info(f"old random: {len(rnd_train)} train / {len(rnd_test)} test hits")
    if args.limit:
        rnd_train, rnd_test = rnd_train[:args.limit], rnd_test[:max(4, args.limit // 4)]
    model = CoilGNN(mesh, args.mp).to(dev)

    def val_onehit(steps):
        return lambda: float(np.mean([onehit_loss(model, mesh, cache, b, dev).item() for b in batches(steps, 8, random.Random(0))]))

    if args.stage1_from:
        # Stage 1 uses only old random data, so it doesn't depend on the split: reuse it.
        src = f"{args.stage1_from}/mp_{args.mp}"
        model = load(mesh, f"{src}/stage1.pt", dev)
        os.makedirs(out, exist_ok=True)
        for f in ("stage1.pt", "stage1_training.json"):
            shutil.copy(f"{src}/{f}", f"{out}/{f}")
        log.info(f"stage 1 reused from {src}")
    else:
        # Stage 1: normalizers learn online while training.
        info1 = train_loop(model, lambda ep: [lambda b=b: onehit_loss(model, mesh, cache, b, dev) for b in batches(rnd_train, 4, rng)],
                           val_onehit(rnd_test), 1e-4, args.epochs1, 20, out, "stage1")
        save(model, f"{out}/stage1.pt", {"stage": 1, **{k: info1[k] for k in ("best_epoch", "best_val_loss")}})

    # Stage 2: old square runs + equal number of stage-1 train hits per epoch.
    model.freeze_normalizers()
    sq = {s: [st for steps in old_steps(run_dir(OLD_SQ, s)).values() for st in steps] for s in TRAIN_SEEDS + [VAL_SEED]}
    sq_train = [st for s in TRAIN_SEEDS for st in sq[s]]
    sq_val = sq[VAL_SEED]
    if args.limit:
        sq_train, sq_val = sq_train[:args.limit], sq_val[:max(4, args.limit // 4)]
    log.info(f"old square: {len(sq_train)} train hits, {len(sq_val)} val hits (seed {VAL_SEED})")

    def items2(ep):
        pool = sq_train + rng.sample(rnd_train, min(len(sq_train), len(rnd_train)))
        return [lambda b=b: onehit_loss(model, mesh, cache, b, dev) for b in batches(pool, 4, rng)]

    info2 = train_loop(model, items2, val_onehit(sq_val), 1e-5, args.epochs2, 20, out, "stage2")
    save(model, f"{out}/stage2.pt", {"stage": 2, **{k: info2[k] for k in ("best_epoch", "best_val_loss")}})


def stage3(args, mesh, cache, dev):
    out = f"{args.out}/mp_{args.mp}"
    rng = random.Random(args.seed)
    model = load(mesh, f"{out}/stage2.pt", dev)
    model.freeze_normalizers()
    cyc = {s: new_cycles(run_dir(args.new_root, s), cache) for s in TRAIN_SEEDS + [VAL_SEED, TEST_SEED]
           if os.path.exists(run_dir(args.new_root, s))}
    train_cycles = [c["steps"] for s in TRAIN_SEEDS if s in cyc for c in cyc[s] if c["steps"]]
    val_cycles = [c["steps"] for c in cyc.get(VAL_SEED, []) if c["complete"] and c["steps"]]
    if args.limit:
        train_cycles, val_cycles = train_cycles[:max(2, args.limit // 6)], val_cycles[:2]
    n_hits = sum(sum(s.hit is not None for s in c) for c in train_cycles)
    rnd_train, rnd_test = load_old_random()
    short = [s for s in rnd_train if s.hit.stroke_mm < 1.0]
    other = [s for s in rnd_train if s.hit.stroke_mm >= 1.0]
    log.info(f"new runs: {len(train_cycles)} train cycles ({n_hits} hits), {len(val_cycles)} val cycles; "
             f"mix-in per epoch: {n_hits // 2} short (<1 mm, pool {len(short)}) + {n_hits - n_hits // 2} other")
    before = evaluate(model, mesh, cache, dev, cyc, rnd_test)

    def items3(ep):
        mix = rng.sample(short, min(n_hits // 2, len(short))) + rng.sample(other, min(n_hits - n_hits // 2, len(other)))
        items = [lambda c=c: chain_loss(model, mesh, cache, c, dev) for c in train_cycles]
        items += [lambda b=b: onehit_loss(model, mesh, cache, b, dev) for b in batches(mix, 4, rng)]
        rng.shuffle(items)
        return items

    val = lambda: float(np.mean([chain_loss(model, mesh, cache, c, dev).item() for c in val_cycles]))
    info3 = train_loop(model, items3, val, 1e-5, args.epochs3, 20, out, "stage3")
    save(model, f"{out}/stage3.pt", {"stage": 3, **{k: info3[k] for k in ("best_epoch", "best_val_loss")}})
    after = evaluate(model, mesh, cache, dev, cyc, rnd_test)
    after["gradient_time_s"] = gradient_time(model, mesh, cache, dev, cyc)
    json.dump({"mp": args.mp, "before_stage3": before, "after_stage3": after,
               "training": {k: info3[k] for k in ("best_epoch", "best_val_loss", "epochs_run", "elapsed_s")}},
              open(f"{out}/stage3_eval.json", "w"), indent=1)
    log.info(f"stage 3 done: {json.dumps(after)[:400]}")


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, mesh, cache, dev, cyc, rnd_test):
    model.eval()
    rest = mesh.rest
    res = {}
    for name, seed in ((f"test_{run_name(TEST_SEED)}", TEST_SEED), (f"val_{run_name(VAL_SEED)}", VAL_SEED)):
        if seed not in cyc:
            continue
        per = {"with_reheat_step": {}, "no_reheat_change": {}}
        for variant, skip in (("with_reheat_step", False), ("no_reheat_change", True)):
            H, C, TA, TD = {}, {}, {}, {}
            for c in cyc[seed]:
                if not c["complete"] or not c["steps"]:
                    continue
                preds, trues = chain(model, mesh, cache, c["steps"], dev, skip_reheat=skip)
                k = 0
                for s, (p, pT), (t, tT) in zip(c["steps"], preds, trues):
                    if s.hit is None:
                        continue
                    k += 1
                    ch, ha = _chamfer_hausdorff_mm(rest + t, rest + p)
                    H.setdefault(k, []).append(ha); C.setdefault(k, []).append(ch)
                    e = (pT - tT).abs()
                    under = mesh.wall & ((rest[:, 0] + t[:, 0] - float(s.hit.center)).abs() <= 0.5 * DIE_W)
                    TA.setdefault(k, []).append(e.mean().item()); TD.setdefault(k, []).append(e[under].mean().item())
            mean = lambda d: float(np.mean([x for v in d.values() for x in v])) if d else None
            per[variant] = {"hausdorff_mm_by_hit_in_cycle": {k: float(np.mean(v)) for k, v in H.items()},
                            "chamfer_mm2_by_hit_in_cycle": {k: float(np.mean(v)) for k, v in C.items()},
                            "temperature_abs_error_C_by_hit_in_cycle": {k: float(np.mean(v)) for k, v in TA.items()},
                            "temperature_abs_error_under_die_C_by_hit_in_cycle": {k: float(np.mean(v)) for k, v in TD.items()},
                            "mean_hausdorff_mm": mean(H), "mean_chamfer_mm2": mean(C),
                            "mean_temperature_abs_error_C": mean(TA), "mean_temperature_abs_error_under_die_C": mean(TD),
                            "n_cycles": len(H.get(1, []))}
        # Reheat step alone: mean per-node error vs assuming no shape change.
        e_step, e_none = [], []
        for c in cyc[seed]:
            for s in c["steps"]:
                if s.hit is None:
                    x0 = cache.get(s.x_in)[0].to(dev); x1 = cache.get(s.x_out)[0].to(dev)
                    p = model.step_shape(features_for(mesh, cache, s, x0, dev)[None], x0[None])[0]
                    e_step.append((p - x1).norm(dim=-1).mean().item()); e_none.append((x0 - x1).norm(dim=-1).mean().item())
        per["reheat_step_mean_node_error_mm"] = float(np.mean(e_step)) if e_step else None
        per["no_change_mean_node_error_mm"] = float(np.mean(e_none)) if e_none else None
        res[name] = per
    # Forgetting check: old random test hits with strokes under 1 mm.
    short = [s for s in rnd_test if s.hit.stroke_mm < 1.0]
    se, seT, n, nT = 0.0, 0.0, 0, 0
    for b in batches(short, 8, random.Random(0)):
        f, x, d = batch_tensors(mesh, cache, b, dev)
        p = model.out_norm.inverse(model.delta_normed(f, x))
        se += ((p[..., :3] - d[..., :3]) ** 2).sum().item(); n += d[..., :3].numel()
        seT += ((p[..., 3] - d[..., 3]) ** 2).sum().item(); nT += d[..., 3].numel()
    res["old_random_test_under_1mm"] = {"n_hits": len(short), "rmse_mm": float(np.sqrt(se / max(n, 1))),
                                        "temperature_rmse_C": float(np.sqrt(seT / max(nT, 1)))}
    return res


def gradient_time(model, mesh, cache, dev, cyc, repeats=10):
    """Wall time of one MPC-style gradient: a reheat step + 6 hits from a test
    cycle's start, cost = sum of final positions, gradient w.r.t. the 18 hit
    controls (die centre, angle, stroke)."""
    c = next((c for c in cyc.get(TEST_SEED, []) if c["complete"] and c["steps"][0].hit is None), None)
    if c is None:
        return None
    model.eval()
    hits = [s for s in c["steps"] if s.hit is not None][:6]
    times = []
    for i in range(repeats + 2):
        ctrl = torch.tensor([[s.hit.center, s.hit.angle_deg, s.hit.stroke_mm] for s in hits], device=dev, requires_grad=True)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        x = cache.get(c["steps"][0].x_in)[0].to(dev)
        x = model.step_shape(features_for(mesh, cache, c["steps"][0], x, dev)[None], x[None])[0]
        T = cache.get(c["steps"][0].x_out)[1].to(dev)
        for j, s in enumerate(hits):
            h = Hit(ctrl[j, 1], ctrl[j, 2], center=ctrl[j, 0])
            xn, Tn = model.step(node_features(mesh, x, T, h)[None], x[None], T[None])
            x, T = xn[0], Tn[0]
        (mesh.rest + x).sum().backward()
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        if i >= 2:
            times.append(time.perf_counter() - t0)
    return float(np.median(times))


def summarize(args):
    rows = {}
    for d in sorted(glob.glob(f"{args.out}/mp_*/stage3_eval.json")):
        e = json.load(open(d))
        rows[e["mp"]] = e
    if not rows:
        log.info("nothing to summarize"); return
    summ = {}
    for m, e in sorted(rows.items()):
        a, b = e["after_stage3"], e["before_stage3"]
        t = next((v for k, v in a.items() if k.startswith("test_")), {})
        summ[m] = {"cycle_chain_mean_hausdorff_mm": t.get("with_reheat_step", {}).get("mean_hausdorff_mm"),
                   "cycle_chain_mean_chamfer_mm2": t.get("with_reheat_step", {}).get("mean_chamfer_mm2"),
                   "cycle_chain_no_reheat_change_hausdorff_mm": t.get("no_reheat_change", {}).get("mean_hausdorff_mm"),
                   "reheat_step_error_mm": t.get("reheat_step_mean_node_error_mm"),
                   "reheat_no_change_error_mm": t.get("no_change_mean_node_error_mm"),
                   "cycle_chain_temperature_abs_error_C": t.get("with_reheat_step", {}).get("mean_temperature_abs_error_C"),
                   "cycle_chain_temperature_abs_error_under_die_C":
                       t.get("with_reheat_step", {}).get("mean_temperature_abs_error_under_die_C"),
                   "short_stroke_temperature_rmse_after_C": a["old_random_test_under_1mm"].get("temperature_rmse_C"),
                   "short_stroke_rmse_before_mm": b["old_random_test_under_1mm"]["rmse_mm"],
                   "short_stroke_rmse_after_mm": a["old_random_test_under_1mm"]["rmse_mm"],
                   "gradient_time_s": a.get("gradient_time_s")}
    json.dump(summ, open(f"{args.out}/summary.json", "w"), indent=1)
    for m, r in summ.items():
        log.info(f"M={m}: {r}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["12", "3"])
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--mp", type=int, default=4)
    p.add_argument("--out", default=OUT)
    p.add_argument("--new-root", default=NEW_SQ)
    p.add_argument("--epochs1", type=int, default=100)
    p.add_argument("--epochs2", type=int, default=300)
    p.add_argument("--epochs3", type=int, default=300)
    p.add_argument("--limit", type=int, default=0, help="Smoke test: cap the number of examples per split.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--test-run", default="square", help="Held-out test run: a jitter seed number, or 'square'.")
    p.add_argument("--stage1-from", default=None, help="Sweep dir whose mp_M/stage1.pt to reuse (stage 12 only).")
    args = p.parse_args()
    set_split(None if args.test_run == "square" else int(args.test_run))
    log.info(f"split: train {TRAIN_SEEDS}, val {VAL_SEED}, test {TEST_SEED}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.summarize:
        return summarize(args)
    dev = args.device
    mesh = Mesh(f"{PRE_DIR}/rollout_01/undeformed.vtu", dev)   # same billet mesh as every dataset
    cache = VtuCache(mesh)
    log.info(f"stage {args.stage}, M={args.mp}, device {dev}")
    (stage12 if args.stage == "12" else stage3)(args, mesh, cache, dev)


if __name__ == "__main__":
    main()
