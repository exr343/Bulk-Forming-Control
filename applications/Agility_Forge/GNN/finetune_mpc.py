"""Finetune the temperature-state coil GNN (coil_T.py's stage3.pt) on the MPC's
own hits, so the surrogate learns the states and actions its planner actually
visits (DAgger-style data aggregation).

Data
  MPC runs   every cycle of each --mpc-runs folder (mpc_coil.py writes the data
             runs' manifest format, so coil_T.new_cycles reads it). Every
             --val-every-th complete cycle of each run is held out for
             validation; the rest train (an unfinished last cycle trains too).
  replay     each epoch, as many 5-pass training cycles (coil_T's train runs:
             seeds 1-8; the plain square stays the test run) as there are MPC
             training cycles, plus stage 3's old-random one-hit mix-in, to keep
             the accuracy on scheduled data.
Loss: coil_T.chain_loss (cycle chains) / onehit_loss, as in stage 3. lr 1e-5,
normalizers frozen. Early stopping on the mean of two validation losses, each
averaged over its cycles: the held-out MPC cycles and seed 9 (stage 3's
validation run).

Evaluation before/after: held-out and training MPC cycles (Hausdorff, Chamfer,
temperature error per hit in the cycle) and coil_T.evaluate (plain square test
run, seed 9, old random short strokes) as the forgetting check.

Usage (from the repo root):
    python -m applications.Agility_Forge.GNN.finetune_mpc
    python -m applications.Agility_Forge.GNN.finetune_mpc --mpc-runs <run dir> ... --out <dir>
"""

import argparse
import json
import logging
import os
import random

import numpy as np
import torch

from applications.Agility_Forge.GNN import coil_T as C
from applications.Agility_Forge.GNN.train import _chamfer_hausdorff_mm

log = logging.getLogger("coil")

AF = C.AF
INIT = f"{AF}/GNN/coil_T_sweep_test_square/mp_5/stage3.pt"
MPC_RUNS = [f"{AF}/control/results/coil_mpc_tstate_M5", f"{AF}/control/results/coil_mpc_tstate_M5_multistart"]
OUT = f"{AF}/GNN/coil_T_mpc_finetune"


def split_mpc(runs, cache, val_every):
    """MPC cycles -> (train, val) lists of Step lists."""
    train, val = [], []
    for run in runs:
        k = 0
        for c in C.new_cycles(run, cache):
            if not c["steps"]:
                continue
            if c["complete"]:
                k += 1
                if k % val_every == 0:
                    val.append(c["steps"])
                    continue
            train.append(c["steps"])
    return train, val


@torch.no_grad()
def cycle_metrics(model, mesh, cache, dev, cycles):
    """Chain each cycle from its true start; Hausdorff, Chamfer and temperature
    error of every hit, by hit number in the cycle and overall."""
    model.eval()
    H, Ch, TA = {}, {}, {}
    for steps in cycles:
        preds, trues = C.chain(model, mesh, cache, steps, dev)
        k = 0
        for s, (p, pT), (t, tT) in zip(steps, preds, trues):
            if s.hit is None:
                continue
            k += 1
            ch, ha = _chamfer_hausdorff_mm(mesh.rest + t, mesh.rest + p)
            H.setdefault(k, []).append(ha); Ch.setdefault(k, []).append(ch)
            TA.setdefault(k, []).append((pT - tT).abs().mean().item())
    mean = lambda d: float(np.mean([x for v in d.values() for x in v])) if d else None
    return {"n_cycles": len(cycles), "mean_hausdorff_mm": mean(H), "mean_chamfer_mm2": mean(Ch),
            "mean_temperature_abs_error_C": mean(TA),
            "hausdorff_mm_by_hit_in_cycle": {k: float(np.mean(v)) for k, v in H.items()},
            "chamfer_mm2_by_hit_in_cycle": {k: float(np.mean(v)) for k, v in Ch.items()},
            "temperature_abs_error_C_by_hit_in_cycle": {k: float(np.mean(v)) for k, v in TA.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", default=INIT, help="Checkpoint to start from.")
    p.add_argument("--mpc-runs", nargs="+", default=MPC_RUNS, help="MPC run folders (missing ones are skipped).")
    p.add_argument("--out", default=OUT)
    p.add_argument("--val-every", type=int, default=4, help="Hold out every n-th complete MPC cycle of each run.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--limit", action="store_true", help="Smoke test: a few cycles and hits only.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    C.set_split(None)                                  # plain square = test, seed 9 = validation, seeds 1-8 train
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = random.Random(args.seed)
    dev = args.device

    mesh = C.Mesh(f"{C.PRE_DIR}/rollout_01/undeformed.vtu", dev)
    cache = C.VtuCache(mesh)
    model = C.load(mesh, args.init, dev)
    model.freeze_normalizers()
    out = f"{args.out}/mp_{model.mp_steps}"

    runs = [r for r in args.mpc_runs if os.path.exists(f"{r}/manifest.json")]
    mpc_train, mpc_val = split_mpc(runs, cache, args.val_every)
    cyc = {s: C.new_cycles(C.run_dir(C.NEW_SQ, s), cache) for s in C.TRAIN_SEEDS + [C.VAL_SEED, C.TEST_SEED]}
    replay_pool = [c["steps"] for s in C.TRAIN_SEEDS for c in cyc[s] if c["steps"]]
    sched_val = [c["steps"] for c in cyc[C.VAL_SEED] if c["complete"] and c["steps"]]
    rnd_train, rnd_test = C.load_old_random()
    short = [s for s in rnd_train if s.hit.stroke_mm < 1.0]
    other = [s for s in rnd_train if s.hit.stroke_mm >= 1.0]
    if args.limit:
        mpc_train, mpc_val, sched_val, replay_pool = mpc_train[:2], mpc_val[:1], sched_val[:1], replay_pool[:4]
        cyc = {s: v[:2] for s, v in cyc.items()}
        rnd_test = [s for s in rnd_test if s.hit.stroke_mm < 1.0][:8]
    n_mpc = sum(s.hit is not None for c in mpc_train for s in c)
    log.info(f"MPC runs {runs}: {len(mpc_train)} train cycles ({n_mpc} hits), {len(mpc_val)} val cycles; "
             f"replay pool {len(replay_pool)} cycles; seed {C.VAL_SEED} val {len(sched_val)} cycles")

    def evaluate():
        r = {"mpc_val": cycle_metrics(model, mesh, cache, dev, mpc_val),
             "mpc_train": cycle_metrics(model, mesh, cache, dev, mpc_train)}
        r.update(C.evaluate(model, mesh, cache, dev, cyc, rnd_test))
        return r

    before = evaluate()
    log.info(f"before: MPC val {json.dumps({k: v for k, v in before['mpc_val'].items() if k.startswith('mean')})}")

    def items(ep):
        rep = rng.sample(replay_pool, min(len(mpc_train), len(replay_pool)))
        n_hits = sum(s.hit is not None for c in mpc_train + rep for s in c)
        mix = rng.sample(short, min(n_hits // 2, len(short))) + rng.sample(other, min(n_hits - n_hits // 2, len(other)))
        its = [lambda c=c: C.chain_loss(model, mesh, cache, c, dev) for c in mpc_train + rep]
        its += [lambda b=b: C.onehit_loss(model, mesh, cache, b, dev) for b in C.batches(mix, 4, rng)]
        rng.shuffle(its)
        return its

    vloss = lambda cs: float(np.mean([C.chain_loss(model, mesh, cache, c, dev).item() for c in cs]))
    val = lambda: float(np.mean([vloss(cs) for cs in (mpc_val, sched_val) if cs]))
    info = C.train_loop(model, items, val, args.lr, args.epochs, args.patience, out, "finetune_mpc")
    C.save(model, f"{out}/finetune_mpc.pt", {"stage": "finetune_mpc", "init": args.init, "mpc_runs": runs,
                                             **{k: info[k] for k in ("best_epoch", "best_val_loss")}})
    after = evaluate()
    json.dump({"mp": model.mp_steps, "args": vars(args), "mpc_runs": runs,
               "n_cycles": {"mpc_train": len(mpc_train), "mpc_val": len(mpc_val), "mpc_train_hits": n_mpc,
                            "replay_pool": len(replay_pool), "seed9_val": len(sched_val)},
               "before": before, "after": after,
               "training": {k: info[k] for k in ("best_epoch", "best_val_loss", "epochs_run", "elapsed_s")}},
              open(f"{out}/finetune_mpc_eval.json", "w"), indent=1)
    log.info(f"after: MPC val {json.dumps({k: v for k, v in after['mpc_val'].items() if k.startswith('mean')})}")


if __name__ == "__main__":
    main()
