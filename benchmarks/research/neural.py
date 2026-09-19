"""Pinned, local, non-truncating cross-encoder baseline; no benchmark labels enter it."""

import json
from time import perf_counter, process_time

from benchmarks.public.common import digest, read_json, write_json
from benchmarks.research.design import inputs, policies, score_path


def run(split):
    import torch
    from sentence_transformers import CrossEncoder

    protocol, datasets = inputs()
    if split == "evaluation":
        policies(split, protocol)  # Require freeze before any evaluation inference.
    config = protocol["neural"]
    torch.set_num_threads(config["threads"])
    model = CrossEncoder(
        config["model"],
        revision=config["revision"],
        device="cpu",
        max_length=config["max_length"],
        model_kwargs={"attn_implementation": config["attention"]},
    )
    model.predict([("Warmup", "Warmup passage")], show_progress_bar=False)
    for data in datasets.values():
        for case in data[split]:
            path = score_path(split, case, "ettin")
            pairs = [(case["query"], d["text"]) for d in case["documents"]]
            fingerprint = digest({"pairs": pairs, "config": config})
            if path.exists():
                if read_json(path)["fingerprint"] != fingerprint:
                    raise ValueError("Neural cache fingerprint mismatch")
                continue
            lengths = [len(model.tokenizer(q, p, truncation=False)["input_ids"]) for q, p in pairs]
            if max(lengths) > config["max_length"]:
                raise ValueError(
                    "Baseline would truncate; stop rather than silently discard evidence"
                )
            started, cpu = perf_counter(), process_time()
            scores = model.predict(
                pairs, batch_size=config["batch_size"], show_progress_bar=False
            ).tolist()
            result = {
                "fingerprint": fingerprint,
                "scores": scores,
                "model": config["model"],
                "revision": config["revision"],
                "elapsed_ms": (perf_counter() - started) * 1000,
                "cpu_seconds": process_time() - cpu,
                "pair_token_lengths": lengths,
                "truncated_pairs": 0,
                "api_cost_usd": 0,
                "total_cost_usd": None,
            }
            write_json(path, result)
            print(
                json.dumps({"scored": case["id"], "scorer": "ettin", "ms": result["elapsed_ms"]}),
                flush=True,
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=["development", "evaluation"])
    run(parser.parse_args().split)
