#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

PROMPT_TEMPLATE = """
    目的: 放射線科専門医が作成した参照放射線診断レポートと比較して、候補レポートの正確性を評価すること。

    プロセスの概要: 以下が提示される。

    1. 判断基準。
    2. 参照放射線診断レポート。
    3. 候補放射線診断レポート。
    4. 評価のための出力形式。

    1. 判断基準:

    各候補レポートについて、次を判定すること。

    臨床的に有意な誤りの件数。
    臨床的に重要でない誤りの件数。

    誤りは次のいずれかのカテゴリーに該当する。

    a) 候補報告における虚偽の所見報告。
    b) 参照に存在する所見の欠落。
    c) 所見の解剖学的位置/位置関係の誤同定。
    d) 所見の重症度の誤評価。
    e) 参照にない比較の記載。
    f) 以前の検査からの変化を示す比較の省略。
    注: レポートの文体ではなく臨床所見に着目すること。両方のレポートに現れる所見のみを評価すること。

    2. 参照レポート:
    __REFERENCE_REPORT__

    3. 候補レポート:
    __CANDIDATE_REPORT__

    4. 評価の報告方法:

    エラーがない場合でも、以下の特定の形式に従って出力すること。
    ```
    [説明]:
    <説明>

    [臨床的に有意な誤り]:
    (a) <誤りの種類>: <誤りの数>. <誤り1>; <誤り2>; ...; <誤りn>
    ....
    (f) <誤りの種類>: <誤りの数>. <誤り1>; <誤り2>; ...; <誤りn>

    [臨床的に重要でない誤り]:
    (a) <誤りの種類>: <誤りの数>. <誤り1>; <誤り2>; ...; <誤りn>
    ....
    (f) <誤りの種類>: <誤りの数>. <誤り1>; <誤り2>; ...; <誤りn>

    [一致する所見]:
    <一致した所見の数>. <所見1>; <所見2>; ...; <所見n>
    ```
    """.lstrip("\n")

INSTRUCTIONS = """
Translate the supplied GREEN dataset records from English to Japanese.
Keep all JSON key names unchanged in English. Translate only key.candidate,
key.reference, and response. Preserve record indices, medical meaning,
laterality, negation, uncertainty, measurements, severity, temporal change,
line breaks, and anonymization placeholders such as ___, ____, XXXX, and xxxx.
Do not correct source errors because they are part of the evaluation dataset.
Translate response headings as follows when present:
[Explanation] -> [説明]
[Clinically Significant Errors] -> [臨床的に有意な誤り]
[Clinically Insignificant Errors] -> [臨床的に重要でない誤り]
[Matched Findings] -> [一致する所見]
Return only JSON matching the required schema.
""".strip()

ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "index": {"type": "integer"},
        "key": {
            "type": "object",
            "properties": {
                "candidate": {"type": "string"},
                "reference": {"type": "string"},
            },
            "required": ["candidate", "reference"],
            "additionalProperties": False,
        },
        "response": {"type": "string"},
    },
    "required": ["index", "key", "response"],
    "additionalProperties": False,
}

BATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"records": {"type": "array", "items": ITEM_SCHEMA}},
    "required": ["records"],
    "additionalProperties": False,
}

_tls = threading.local()


def get_client() -> OpenAI:
    if not hasattr(_tls, "client"):
        _tls.client = OpenAI()
    return _tls.client


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--errors", type=Path)
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--limit", type=int)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Top-level JSON must be a list.")
    return data


def make_api_item(index: int, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": index,
        "key": {
            "candidate": record["key"]["candidate"],
            "reference": record["key"]["reference"],
        },
        "response": record["response"],
    }


def build_prompt(reference: str, candidate: str) -> str:
    return (PROMPT_TEMPLATE
            .replace("__REFERENCE_REPORT__", reference)
            .replace("__CANDIDATE_REPORT__", candidate))


def finalize(item: dict[str, Any]) -> dict[str, Any]:
    candidate = item["key"]["candidate"]
    reference = item["key"]["reference"]
    return {
        "key": {"candidate": candidate, "reference": reference},
        "response": item["response"],
        "prompt": build_prompt(reference, candidate),
    }


def placeholder_counts(text: str) -> dict[str, int]:
    return {t: text.count(t) for t in ("___", "____", "XXXX", "xxxx")}


def validate_batch(src: list[dict[str, Any]], dst: list[dict[str, Any]]) -> None:
    if len(src) != len(dst):
        raise ValueError("Batch length changed.")
    src_map = {x["index"]: x for x in src}
    dst_map = {x["index"]: x for x in dst}
    if set(src_map) != set(dst_map):
        raise ValueError("Record indices changed.")
    for i, s in src_map.items():
        d = dst_map[i]
        for field in ("candidate", "reference"):
            if placeholder_counts(s["key"][field]) != placeholder_counts(d["key"][field]):
                raise ValueError(f"Placeholder mismatch at record {i}, {field}.")
        if placeholder_counts(s["response"]) != placeholder_counts(d["response"]):
            raise ValueError(f"Placeholder mismatch at record {i}, response.")


def translate_batch(items: list[dict[str, Any]], model: str, max_retries: int) -> list[dict[str, Any]]:
    payload = json.dumps({"records": items}, ensure_ascii=False, separators=(",", ":"))
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            r = get_client().responses.create(
                model=model,
                reasoning={"effort": "minimal"},
                instructions=INSTRUCTIONS,
                input=payload,
                text={"format": {
                    "type": "json_schema",
                    "name": "green_translation_batch",
                    "schema": BATCH_SCHEMA,
                    "strict": True,
                }},
            )
            translated = json.loads(r.output_text)["records"]
            validate_batch(items, translated)
            by_index = {x["index"]: x for x in translated}
            return [by_index[x["index"]] for x in items]
        except Exception as exc:
            last = exc
            if attempt == max_retries:
                break
            time.sleep(min(60, 2 ** attempt + random.random()))
    raise RuntimeError(f"Batch failed after {max_retries + 1} attempts") from last


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    done: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                done[int(row["index"])] = row["record"]
    return done


def chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[i:i + size] for i in range(0, len(values), size)]


def main() -> int:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2
    if args.batch_size < 1 or args.workers < 1:
        raise ValueError("--batch-size and --workers must be >= 1")

    checkpoint = args.checkpoint or args.output.with_name(args.output.name + ".checkpoint.jsonl")
    errors = args.errors or args.output.with_name(args.output.name + ".errors.jsonl")

    if args.overwrite:
        for p in (args.output, checkpoint, errors):
            if p.exists():
                p.unlink()

    source = load_json(args.input)
    stop = len(source) if args.limit is None else min(len(source), args.start_index + args.limit)
    selected = list(range(args.start_index, stop))
    completed = read_checkpoint(checkpoint)
    pending = [i for i in selected if i not in completed]
    batches = chunks(pending, args.batch_size)

    print(f"Selected: {len(selected)}, completed: {len(selected)-len(pending)}, pending: {len(pending)}")
    print(f"batch-size={args.batch_size}, workers={args.workers}, model={args.model}")

    failed: list[list[int]] = []
    with tqdm(total=len(selected), initial=len(selected) - len(pending), unit="record") as bar:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for idxs in batches:
                api_items = [make_api_item(i, source[i]) for i in idxs]
                fut = pool.submit(translate_batch, api_items, args.model, args.max_retries)
                futures[fut] = idxs

            for fut in as_completed(futures):
                idxs = futures[fut]
                try:
                    translated = fut.result()
                    rows = []
                    for item in translated:
                        i = item["index"]
                        record = finalize(item)
                        completed[i] = record
                        rows.append({"index": i, "record": record})
                    append_jsonl(checkpoint, rows)
                    bar.update(len(idxs))
                except Exception as exc:
                    failed.append(idxs)
                    append_jsonl(errors, [{"indices": idxs, "error": str(exc)}])
                    print(f"\nFailed batch {idxs}: {exc}", file=sys.stderr)

    if failed:
        print("Some records failed. Rerun the same command to retry unfinished records.", file=sys.stderr)
        return 1

    missing = [i for i in selected if i not in completed]
    if missing:
        raise RuntimeError(f"Missing records: {missing[:10]}")

    tmp = args.output.with_name(args.output.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump([completed[i] for i in selected], f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(args.output)
    print(f"Completed: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
