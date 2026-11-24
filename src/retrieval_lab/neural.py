"""Independent controlled RAE/QRAE training from explicit mathematical losses.

This is the centered/no-bias BEIR comparison, not a claim of reproducing the
released RAE code defaults. The paper-reproduction protocol documents differences.
Import this module only in Torch workers; it has no FAISS dependency.
"""

from __future__ import annotations

import numpy as np


def ranking_loss(query, positive, negative, margin: float):
    """One-negative cosine hinge, averaged over sampled positive pairs."""
    import torch.nn.functional as functional

    query = functional.normalize(query, dim=1)
    positive = functional.normalize(positive, dim=1)
    negative = functional.normalize(negative, dim=1)
    gap = (query * positive).sum(1) - (query * negative).sum(1)
    return functional.relu(margin - gap).mean(), gap


def train_neural(x, dimension, seed, method, config, supervision=None):
    import torch

    from .compressors import select_device

    if method not in {"rae", "qrae", "rank_only"}:
        raise ValueError("Unsupported neural method")
    supervised = method in {"qrae", "rank_only"}
    if supervised and supervision is None:
        raise ValueError("Training query supervision is required")
    if not supervised and supervision is not None:
        raise ValueError("RAE must not receive query supervision")
    device = select_device(config["device"])
    torch.set_num_threads(config.get("threads", 4))
    torch.manual_seed(seed)
    # Separate RNG streams keep reconstruction batches identical across methods.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sampling_rng = np.random.default_rng(seed + 104729)
    mean = (
        x.mean(axis=0, dtype=np.float64).astype(np.float32)
        if config["center"]
        else np.zeros(x.shape[1], dtype=np.float32)
    )
    inputs = torch.from_numpy(np.ascontiguousarray(x - mean)).to(device)
    encoder = torch.nn.Linear(x.shape[1], dimension, bias=False).to(device)
    # Construct both layers even for rank-only to keep encoder initialization fixed.
    decoder = torch.nn.Linear(dimension, x.shape[1], bias=False).to(device)
    parameters = list(encoder.parameters())
    if method != "rank_only":
        parameters += list(decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config["lr"], weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config["steps"] - 1), eta_min=config["min_lr"]
    )
    regularization_weight = 0.0 if method == "rank_only" else config["regularization_weight"]
    ranking_weight = config["ranking_weight"] if method == "qrae" else 1.0
    if supervised:
        from .supervision import sample_triplets

        if supervision["documents"].shape != x.shape or not np.allclose(
            supervision["documents"], x, atol=1e-6
        ):
            raise ValueError("Supervision document rows differ from compressor fit corpus")
        query_inputs = torch.from_numpy(np.ascontiguousarray(supervision["queries"] - mean)).to(
            device
        )
        # Fixed train-only probe; it is not a held-out validation result.
        probe_rng = np.random.default_rng(seed + 130363)
        fixed_triplet = sample_triplets(
            supervision,
            probe_rng,
            min(256, config["batch_size"]),
            config["negative_strategy"],
        )
        fixed_triplet = tuple(torch.as_tensor(a, device=device) for a in fixed_triplet)
    probe = inputs[: min(1024, len(inputs))]
    history = []
    for step in range(1, config["steps"] + 1):
        ids = torch.randint(len(inputs), (config["batch_size"],), generator=generator)
        batch = inputs[ids.to(device)]
        optimizer.zero_grad(set_to_none=True)
        if method != "rank_only":
            latent = encoder(batch)
            reconstruction = (decoder(latent) - batch).square().sum(1).mean()
            regularizer = encoder.weight.square().sum() + decoder.weight.square().sum()
            loss = reconstruction + regularization_weight * regularizer
        else:
            reconstruction = inputs.new_tensor(0.0)
            regularizer = inputs.new_tensor(0.0)
            loss = inputs.new_tensor(0.0)
        rank = inputs.new_tensor(0.0)
        active = inputs.new_tensor(0.0)
        if supervised:
            qi, pi, ni = sample_triplets(
                supervision, sampling_rng, config["batch_size"], config["negative_strategy"]
            )
            qi, pi, ni = (torch.as_tensor(a, device=device) for a in (qi, pi, ni))
            rank, gaps = ranking_loss(
                encoder(query_inputs[qi]),
                encoder(inputs[pi]),
                encoder(inputs[ni]),
                config["margin"],
            )
            active = (gaps < config["margin"]).float().mean()
            loss = loss + ranking_weight * rank
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{method} non-finite loss at step {step}")
        loss.backward()
        # Observe gradients without clipping, preserving the declared objective.
        grad_norm = torch.sqrt(sum(p.grad.square().sum() for p in parameters if p.grad is not None))
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"{method} non-finite gradient at step {step}")
        optimizer.step()
        used_lr = optimizer.param_groups[0]["lr"]
        if step < config["steps"]:
            scheduler.step()
        if step == 1 or step % config["log_every"] == 0 or step == config["steps"]:
            with torch.no_grad():
                encoded_probe = encoder(probe)
                norms = torch.linalg.vector_norm(encoded_probe, dim=1)
                if not torch.isfinite(norms).all() or norms.min() < 1e-12:
                    raise FloatingPointError("Collapsed or non-finite corpus probe latent")
                probe_recon = (
                    (decoder(encoded_probe) - probe).square().sum(1).mean().item()
                    if method != "rank_only"
                    else 0.0
                )
                train_probe_rank = 0.0
                if supervised:
                    pqi, ppi, pni = fixed_triplet
                    train_probe_rank = ranking_loss(
                        encoder(query_inputs[pqi]),
                        encoder(inputs[ppi]),
                        encoder(inputs[pni]),
                        config["margin"],
                    )[0].item()
            history.append(
                {
                    "step": step,
                    "total_loss": loss.item(),
                    "batch_reconstruction": reconstruction.item(),
                    "regularizer": regularizer.item(),
                    "weighted_regularizer": regularization_weight * regularizer.item(),
                    "batch_ranking": rank.item(),
                    "weighted_ranking": ranking_weight * rank.item(),
                    "active_triplet_fraction": active.item(),
                    "corpus_probe_reconstruction": probe_recon,
                    "train_probe_ranking": train_probe_rank,
                    "gradient_norm": grad_norm.item(),
                    "latent_norm_min": norms.min().item(),
                    "latent_norm_mean": norms.mean().item(),
                    "learning_rate": used_lr,
                }
            )
            print(
                f"  {method} seed={seed} d={dimension} step={step} "
                f"rec={probe_recon:.6f} rank={train_probe_rank:.6f}",
                flush=True,
            )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    arrays = {"mean": mean, "encoder": encoder.weight.detach().cpu().numpy().copy()}
    if method != "rank_only":
        arrays["decoder"] = decoder.weight.detach().cpu().numpy().copy()
    singular = np.linalg.svd(arrays["encoder"].astype(np.float64), compute_uv=False)
    info = {
        "architecture": (
            "shared linear encoder only; no bias"
            if method == "rank_only"
            else "shared linear encoder/decoder; no bias; untied weights"
        ),
        "centering": "corpus mean" if config["center"] else "none",
        "definition": "controlled RAE-style BEIR comparison; see reproduction protocol",
        "reconstruction_loss": "batch mean of dimension-summed squared errors; raw latent",
        "regularizer": "sum of squared encoder and decoder weights; explicit coefficient",
        "ranking_loss": "one-negative normalized-cosine hinge averaged over positive pairs",
        "regularization_weight": regularization_weight,
        "ranking_weight": ranking_weight if supervised else 0.0,
        "optimizer": "Adam; weight_decay=0; cosine schedule per update; no clipping",
        "training": dict(config),
        "actual_device": device,
        "actual_steps": config["steps"],
        "history": history,
        "checkpoint_selection": "last fixed step; no test access; dev comparison external",
        "supervision": "train queries/qrels only" if supervised else "none",
        "encoder_singular_values": singular.tolist(),
        "encoder_condition_number": float(singular[0] / max(singular[-1], 1e-30)),
        "probe_scope": "training-corpus reconstruction and fixed train triplets, not validation",
    }
    return arrays, info
