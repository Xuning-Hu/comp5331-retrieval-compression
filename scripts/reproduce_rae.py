"""Independent RAE geometry experiment; see docs/rae-reproduction-protocol.md.

Accepts existing vectors, never downloads data or imports the authors' code.
The two training conventions are deliberately separate. Results on replacement
embeddings are algorithm checks, not a reproduction of published table numbers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

AUTHOR_COMMIT = "138b00f19033121cec947be28a41c9a6bf0892e3"


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lengths = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or np.any(lengths < 1e-12):
        raise ValueError("Cosine evaluation requires finite, nonzero vectors")
    return np.ascontiguousarray(x / lengths)


def nearest_neighbors(base, queries, k, metric, self_positions=None, block_size=128):
    """Exact search, with deterministic ties resolved by ascending base row.

    self_positions explicitly maps each query to its identical base row, or -1.
    Removing the first result would incorrectly handle duplicate embeddings.
    """
    base = np.asarray(base, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    if metric not in {"cosine", "euclidean"}:
        raise ValueError("Unknown metric")
    if base.ndim != 2 or queries.ndim != 2 or base.shape[1] != queries.shape[1]:
        raise ValueError("Incompatible vector matrices")
    if not np.isfinite(base).all() or not np.isfinite(queries).all():
        raise ValueError("Non-finite vectors")
    if self_positions is None:
        self_positions = np.full(len(queries), -1, dtype=np.int64)
    self_positions = np.asarray(self_positions, dtype=np.int64)
    if len(self_positions) != len(queries):
        raise ValueError("self_positions length mismatch")
    if np.any(self_positions < -1) or np.any(self_positions >= len(base)):
        raise ValueError("Invalid self position")
    available = len(base) - int(np.any(self_positions >= 0))
    if not 1 <= k <= available or block_size < 1:
        raise ValueError("Not enough candidates for the requested k")
    if metric == "cosine":
        base, queries = normalize(base), normalize(queries)
    base_norms = np.einsum("ij,ij->i", base, base)
    answer = np.empty((len(queries), k), dtype=np.int64)
    for start in range(0, len(queries), block_size):
        query_block = queries[start : start + block_size]
        scores = query_block @ base.T
        if metric == "euclidean":
            scores *= 2
            scores -= base_norms[None, :]
            scores -= np.einsum("ij,ij->i", query_block, query_block)[:, None]
        positions = self_positions[start : start + len(query_block)]
        valid = np.flatnonzero(positions >= 0)
        scores[valid, positions[valid]] = -np.inf
        answer[start : start + len(query_block)] = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return answer


def overlaps(reference, reduced, ks):
    if reference.shape != reduced.shape or not len(reference):
        raise ValueError("Neighbor arrays must have the same nonempty shape")
    answer = {}
    for k in ks:
        if not 1 <= k <= reference.shape[1]:
            raise ValueError("Invalid overlap cutoff")
        values = [len(set(a[:k]) & set(b[:k])) / k for a, b in zip(reference, reduced, strict=True)]
        answer[f"nn_overlap@{k}"] = float(np.mean(values))
    return answer


def split_vectors(vectors, seed, train_fraction=0.9):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between zero and one")
    order = np.random.default_rng(seed).permutation(len(vectors))
    cutoff = int(len(vectors) * train_fraction)
    if cutoff < 2 or cutoff == len(vectors):
        raise ValueError("Need at least two training and one held-out vector")
    return order[:cutoff], order[cutoff:]


def training_budget(sample_count, batch_size, requested_steps, mode):
    batches = math.ceil(sample_count / batch_size)
    if mode == "released":
        epochs = max(1, requested_steps // batches)
        return epochs * batches, epochs, batches
    if mode != "formula":
        raise ValueError("mode must be formula or released")
    return requested_steps, math.ceil(requested_steps / batches), batches


def train_rae(
    x,
    dimension,
    seed,
    mode,
    coefficient,
    steps=3000,
    batch_size=128,
    lr=1e-3,
    min_lr=1e-5,
    device="cpu",
    threads=4,
    log_every=100,
):
    """Train from the mathematical objective using standard Torch building blocks.

    coefficient is a nominal coupled-Adam decay w in both tracks. The formula
    track uses explicit (D*w/2) * (||E||^2 + ||D||^2), accompanying reconstruction
    summed over dimensions. This rescales element-MSE plus w/2 L2 by D, rather
    than accidentally changing relative regularization when changing reduction.
    Bias, batching order, epsilon effects and scheduling still differ by track.
    """
    import torch

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all() or not len(x):
        raise ValueError("Training vectors must be a finite nonempty matrix")
    if not 1 <= dimension <= x.shape[1] or coefficient < 0:
        raise ValueError("Invalid dimension or regularization coefficient")
    if min(steps, batch_size, threads, log_every) < 1 or not 0 <= min_lr <= lr:
        raise ValueError("Invalid training configuration")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    source = torch.from_numpy(np.ascontiguousarray(x)).to(device)
    with_bias = mode == "released"
    encoder = torch.nn.Linear(x.shape[1], dimension, bias=with_bias).to(device)
    decoder = torch.nn.Linear(dimension, x.shape[1], bias=with_bias).to(device)
    parameters = [*encoder.parameters(), *decoder.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=coefficient if with_bias else 0.0)
    actual_steps, epochs, batches = training_budget(len(x), batch_size, steps, mode)
    explicit_coefficient = x.shape[1] * coefficient / 2 if mode == "formula" else 0.0
    order_rng = np.random.default_rng(seed + 1)
    probe = source[: min(len(source), 1024)]
    with torch.no_grad():
        initial_reconstruction = float((decoder(encoder(probe)) - probe).square().sum(1).mean())
    history = []
    step = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        order = order_rng.permutation(len(source))
        for first in range(0, len(order), batch_size):
            if step == actual_steps:
                break
            fraction = epoch / epochs if mode == "released" else step / max(1, actual_steps - 1)
            used_lr = min_lr + (lr - min_lr) * (1 + math.cos(math.pi * fraction)) / 2
            for group in optimizer.param_groups:
                group["lr"] = used_lr
            batch_ids = torch.as_tensor(order[first : first + batch_size], device=device)
            batch = source[batch_ids]
            optimizer.zero_grad(set_to_none=True)
            error = decoder(encoder(batch)) - batch
            reconstruction = error.square().sum(1).mean()
            weights_squared = encoder.weight.square().sum() + decoder.weight.square().sum()
            if mode == "formula":
                penalty = explicit_coefficient * weights_squared
                loss = reconstruction + penalty
            else:
                loss = reconstruction / x.shape[1]
                penalty = coefficient / 2 * sum(p.square().sum() for p in parameters)
            if not torch.isfinite(loss) or not torch.isfinite(penalty):
                raise FloatingPointError(f"Non-finite objective at update {step + 1}")
            loss.backward()
            optimizer.step()
            step += 1
            if step == 1 or step % log_every == 0 or step == actual_steps:
                with torch.no_grad():
                    probe_error = (decoder(encoder(probe)) - probe).square().sum(1).mean()
                history.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "learning_rate": used_lr,
                        "batch_reconstruction_per_vector": float(reconstruction.detach()),
                        "objective_reconstruction": float(loss.detach() - penalty.detach())
                        if mode == "formula"
                        else float(loss.detach()),
                        "regularizer_penalty": float(penalty.detach()),
                        "regularizer_applied_via_optimizer": mode == "released",
                        "probe_reconstruction_per_vector": float(probe_error),
                    }
                )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    if device == "mps":
        torch.mps.synchronize()
    arrays = {
        "encoder": encoder.weight.detach().cpu().numpy().copy(),
        "decoder": decoder.weight.detach().cpu().numpy().copy(),
    }
    if with_bias:
        arrays["encoder_bias"] = encoder.bias.detach().cpu().numpy().copy()
        arrays["decoder_bias"] = decoder.bias.detach().cpu().numpy().copy()
    singular = np.linalg.svd(arrays["encoder"].astype(np.float64), compute_uv=False)
    info = {
        "mode": mode,
        "dimension": dimension,
        "seed": seed,
        "nominal_adam_decay": coefficient,
        "explicit_frobenius_coefficient": explicit_coefficient,
        "centering": "none",
        "bias": with_bias,
        "reconstruction_reduction": "mean over all elements"
        if with_bias
        else "mean of per-vector summed squared errors",
        "optimizer": "Adam; coupled decay on all weights and biases"
        if with_bias
        else "Adam without optimizer decay; explicit L2 on encoder and decoder weights",
        "batch_order": "private NumPy Generator(seed+1) shuffles without replacement each epoch",
        "requested_steps": steps,
        "actual_steps": step,
        "batch_size": batch_size,
        "batches_per_epoch": batches,
        "lr": lr,
        "min_lr": min_lr,
        "schedule": "cosine by epoch" if with_bias else "cosine by update",
        "checkpoint_selection": "final fixed update; no held-out score selection",
        "initial_probe_reconstruction_per_vector": initial_reconstruction,
        "final_probe_reconstruction_per_vector": history[-1]["probe_reconstruction_per_vector"],
        "singular_max": float(singular[0]),
        "singular_min": float(singular[-1]),
        "condition_number_nonzero_spectrum": float(singular[0] / singular[-1])
        if singular[-1] > 1e-12
        else None,
        "encoder_frobenius_squared": float(np.sum(singular**2)),
        "fit_seconds": time.perf_counter() - started,
        "history": history,
    }
    return arrays, info


def project(x, arrays):
    x = np.asarray(x, dtype=np.float32)
    if "mean" in arrays:
        x = x - arrays["mean"]
    result = x @ arrays["encoder"].T
    if "encoder_bias" in arrays:
        result = result + arrays["encoder_bias"]
    return np.ascontiguousarray(result, dtype=np.float32)


def evaluate_projection(arrays, base, heldout, anchor_ids, metric, references, ks):
    # The released RAE pipeline normalizes before the encoder for cosine search.
    base_in, heldout_in = (
        (normalize(base), normalize(heldout)) if metric == "cosine" else (base, heldout)
    )
    base_z, query_z = project(base_in, arrays), project(heldout_in, arrays)
    train_z = base_z[anchor_ids]
    result = {}
    for label, query, self_ids in [
        ("heldout", query_z, None),
        ("train_probe", train_z, anchor_ids),
    ]:
        neighbors = nearest_neighbors(base_z, query, max(ks), metric, self_ids)
        result.update(
            {
                f"{label}_{key}": value
                for key, value in overlaps(references[label], neighbors, ks).items()
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-label", default="replacement-vectors")
    parser.add_argument("--dimensions", nargs="+", type=int, default=[128])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[0.0, 2e-5, 2e-4],
        help="Nominal coupled-Adam decay w; formula L2 coefficient is D*w/2",
    )
    parser.add_argument(
        "--modes", nargs="+", choices=["formula", "released"], default=["formula", "released"]
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--train-probes", type=int, default=256)
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20, 50])
    parser.add_argument(
        "--normalize-training",
        action="store_true",
        help="Optional explicit departure from released default raw training",
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty, to avoid mixing experiment runs")
    if args.train_probes < 1 or min(args.ks) < 1 or min(args.lambdas) < 0:
        parser.error("Invalid probe count, k, or regularization coefficient")
    vectors = np.load(args.vectors, allow_pickle=False)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        parser.error("--vectors must contain a finite two-dimensional NPY array")
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if min(args.dimensions) < 1 or max(args.dimensions) > vectors.shape[1]:
        parser.error("Output dimensions must be between 1 and the input width")
    args.output.mkdir(parents=True, exist_ok=True)
    import torch
    from threadpoolctl import threadpool_limits

    threadpool_limits(args.threads)
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config.update(
        {
            "source_vectors_sha256": digest_file(args.vectors),
            "source_shape": list(vectors.shape),
            "script_sha256": digest_file(Path(__file__)),
            "author_reference_commit": AUTHOR_COMMIT,
            "claim_scope": "Independent algorithm reproduction on supplied vectors; NOT published table replication",
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "precision": "float32 training and search",
            "geometry_protocol": "held-out vectors query training vectors; train probes explicitly exclude self",
            "tie_breaking": "ascending training row position for equal float32 scores",
            "lambda_axis": "nominal coupled-Adam decay w; formula explicit coefficient D*w/2",
        }
    )
    save_json(args.output / "protocol.json", config)
    rows = []
    for seed in args.seeds:
        fit_ids, heldout_ids = split_vectors(vectors, seed, args.train_fraction)
        base, heldout = vectors[fit_ids], vectors[heldout_ids]
        if max(args.ks) >= len(base) or max(args.dimensions) > min(len(base) - 1, base.shape[1]):
            parser.error("Training set is too small for requested cutoffs or centered PCA rank")
        anchors = np.sort(
            np.random.default_rng(seed + 2).choice(
                len(base), min(args.train_probes, len(base)), replace=False
            )
        )
        save_json(
            args.output / f"split-{seed}.json",
            {
                "train_rows": fit_ids.tolist(),
                "heldout_rows": heldout_ids.tolist(),
                "train_probe_positions": anchors.tolist(),
            },
        )
        references = {}
        for metric in ["cosine", "euclidean"]:
            references[metric] = {
                "heldout": nearest_neighbors(base, heldout, max(args.ks), metric),
                "train_probe": nearest_neighbors(
                    base, base[anchors], max(args.ks), metric, anchors
                ),
            }
        # Match baseline preprocessing to each metric, then fit once at max dimension.
        pca_fits = {}
        for metric in references:
            pca_input = normalize(base) if metric == "cosine" else base
            mean = pca_input.mean(axis=0, dtype=np.float64)
            _, _, right = np.linalg.svd(pca_input.astype(np.float64) - mean, full_matrices=False)
            pca_fits[metric] = {
                "mean": mean.astype(np.float32),
                "encoder": right.astype(np.float32),
            }
        for dimension in args.dimensions:
            for metric in references:
                arrays = {**pca_fits[metric], "encoder": pca_fits[metric]["encoder"][:dimension]}
                scores = evaluate_projection(
                    arrays, base, heldout, anchors, metric, references[metric], args.ks
                )
                rows.append(
                    {
                        "method": "pca",
                        "seed": seed,
                        "dimension": dimension,
                        "nominal_adam_decay": 0,
                        "metric": metric,
                        **scores,
                    }
                )
            for mode in args.modes:
                for coefficient in args.lambdas:
                    name = f"{mode}-d{dimension}-s{seed}-w{coefficient:g}"
                    print(f"Training {name}", flush=True)
                    training = normalize(base) if args.normalize_training else base
                    arrays, info = train_rae(
                        training,
                        dimension,
                        seed,
                        mode,
                        coefficient,
                        args.steps,
                        args.batch_size,
                        args.lr,
                        args.min_lr,
                        args.device,
                        args.threads,
                    )
                    folder = args.output / name
                    folder.mkdir()
                    np.savez(folder / "model.npz", **arrays)
                    save_json(folder / "training.json", info)
                    for metric in references:
                        scores = evaluate_projection(
                            arrays, base, heldout, anchors, metric, references[metric], args.ks
                        )
                        row = {
                            "method": mode,
                            "seed": seed,
                            "dimension": dimension,
                            "nominal_adam_decay": coefficient,
                            "metric": metric,
                            "actual_steps": info["actual_steps"],
                            "fit_seconds": info["fit_seconds"],
                            "singular_min": info["singular_min"],
                            "singular_max": info["singular_max"],
                            "condition_number_nonzero_spectrum": info[
                                "condition_number_nonzero_spectrum"
                            ],
                            **scores,
                        }
                        rows.append(row)
                        print(
                            f"  {metric}: heldout NNOverlap@{args.ks[0]}="
                            f"{scores[f'heldout_nn_overlap@{args.ks[0]}']:.4f}",
                            flush=True,
                        )
                    save_json(args.output / "results.json", rows)
    save_json(args.output / "results.json", rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} metric rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
