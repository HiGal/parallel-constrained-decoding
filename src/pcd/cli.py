"""Command line interface.

    pcd models                                   list model aliases
    pcd check MODEL [--backend mlx|torch]        is this model supported? (conformance + tokenization)
    pcd run MODEL PRESET.json                    decode one preset (schema + context)
    pcd run MODEL --schema S.json --context C.txt
    pcd bench MODEL PRESET.json [...]            parallel vs autoregressive latency and agreement
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from .decoding import DecodeOptions


def _load_engine(args: argparse.Namespace):
    from .engine import Engine

    return Engine.load(args.model, args.backend, decode=DecodeOptions(strategy=args.strategy))


def _read_task(args: argparse.Namespace) -> tuple[str, dict]:
    if args.preset:
        p = json.loads(Path(args.preset).read_text())
        return p["context"], p["schema"]
    if not (args.schema and args.context):
        sys.exit("give a preset file, or both --schema and --context")
    return Path(args.context).read_text(), json.loads(Path(args.schema).read_text())


def cmd_models(args: argparse.Namespace) -> None:
    from .models import MODELS

    print(f"{'alias':15} {'params':8} {'mlx':45} {'torch':40} notes")
    for s in MODELS.values():
        print(f"{s.alias:15} {s.params:8} {s.mlx or '-':45} {s.torch or '-':40} {s.notes}")


def cmd_check(args: argparse.Namespace) -> None:
    from .backends import load_backend
    from .compiler import compile_schema
    from .engine import Engine
    from .schema import Schema
    from .testing import check_backend

    backend = load_backend(args.model, args.backend)
    print(json.dumps(backend.describe()))
    checks = check_backend(backend)
    for c in checks:
        print(c)
    engine = Engine(backend)
    demo = Schema.from_dict({
        "is_urgent": {"type": "boolean", "description": "Whether the message is urgent"},
        "sla": {"type": "enum", "choices": ["1_HOUR", "4_HOURS", "12_HOURS", "24_HOURS"], "description": "SLA window"},
        "tier": {"type": "enum", "choices": ["TIER_1", "TIER_2", "SENIOR"], "description": "Support tier"},
    })
    try:
        cs = compile_schema(demo, engine.tokenizer, engine.prompt_format)
        print(f"[PASS] prompt renders ({cs.prompt.mode}); fields tokenize cleanly after the prompt tail")
        out = engine.extract("Server down, customer furious, fix within the hour.", demo)
        print(f"[PASS] end-to-end: {out.values}  ({out.stats.total_ms:.0f} ms, {out.stats.forward_passes} passes)")
        for w in out.warnings:
            print(f"       warning: {w}")
    except Exception as e:
        print(f"[FAIL] schema compilation / extraction: {e}")
        sys.exit(1)
    if not all(c.passed for c in checks):
        sys.exit(1)


def cmd_run(args: argparse.Namespace) -> None:
    engine = _load_engine(args)
    context, schema = _read_task(args)
    out = engine.extract(context, schema)
    if args.json:
        print(json.dumps(out.to_dict(), indent=2, ensure_ascii=False, default=str))
        return
    print(json.dumps(out.values, indent=2, ensure_ascii=False))
    print(f"\n{'field':34} {'value':28} {'prob':>6} {'mass':>6}  method")
    for name, f in out.fields.items():
        mass = f"{f.candidate_mass:.2f}" if f.candidate_mass is not None else "-"
        print(f"{name:34} {str(f.value):28} {f.probability:6.2f} {mass:>6}  {f.method}")
    s = out.stats
    print(
        f"\n{s.total_ms:.0f} ms total | prefill {s.prefill_ms:.0f} ms (+{s.head_prefill_ms:.0f} ms head) | "
        f"decode {s.decode_ms:.0f} ms | {s.forward_passes} forward passes | {s.prompt_tokens} prompt tokens "
        f"({s.cached_tokens} cached)"
    )
    for w in out.warnings:
        print(f"warning: {w}")


def cmd_bench(args: argparse.Namespace) -> None:
    from .baseline import generate_json, lenient_values

    engine = _load_engine(args)
    print(json.dumps(engine.backend.describe()))
    header = f"{'preset':24} {'fields':>6} {'parallel ms':>12} {'passes':>6} {'AR ms':>9} {'AR tok':>7} {'speed-up':>8} {'agree':>7} {'AR schema ok':>12}"
    print(header)
    for path in args.presets:
        p = json.loads(Path(path).read_text())
        schema, context = p["schema"], p["context"]
        engine.extract(context, schema)  # warm: compile, cache head, build kernels
        runs = [engine.extract(context, schema) for _ in range(args.repeats)]
        par = sorted(runs, key=lambda r: r.stats.total_ms)[len(runs) // 2]
        par_ms = statistics.median(r.stats.total_ms for r in runs)
        if args.no_baseline:
            print(f"{Path(path).stem:24} {len(schema):6} {par_ms:12.0f} {par.stats.forward_passes:6}")
            continue
        ar = generate_json(engine, context, schema)
        agree = "-"
        if isinstance(ar.values, dict):
            arv = lenient_values(schema, ar.values)
            agree = f"{sum(1 for k, v in par.values.items() if k in arv and arv[k] == v)}/{len(schema)}"
        print(
            f"{Path(path).stem:24} {len(schema):6} {par_ms:12.0f} {par.stats.forward_passes:6} {ar.total_ms:9.0f} "
            f"{ar.new_tokens:7} {ar.total_ms / par_ms:7.1f}x {agree:>7} {str(ar.schema_match):>12}"
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="pcd", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("model", help="alias (see `pcd models`), Hugging Face repo id or local path")
        p.add_argument("--backend", default="auto", choices=["auto", "mlx", "torch"])
        p.add_argument("--strategy", default="auto", choices=["auto", "exact", "greedy"])

    sub.add_parser("models", help="list model aliases").set_defaults(fn=cmd_models)

    p = sub.add_parser("check", help="check that a model works with pcd")
    common(p)
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("run", help="decode one schema for one context")
    common(p)
    p.add_argument("preset", nargs="?", help="JSON file with 'schema' and 'context'")
    p.add_argument("--schema")
    p.add_argument("--context")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("bench", help="compare against autoregressive JSON generation")
    common(p)
    p.add_argument("presets", nargs="+")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--no-baseline", action="store_true")
    p.set_defaults(fn=cmd_bench)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
