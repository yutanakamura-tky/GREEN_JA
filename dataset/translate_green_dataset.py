#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

PROMPT_TEMPLATE = """
    目的: 放射線科専門医が作成した参照放射線診断
レポートと比較して、候補レポートの正確性を評価すること。

    プロセスの概要: 以下が提示される。

    1. 判断基準。
    2. 参照放射線診断レポート。
    3. 候補放射線診断レポート。
    4. 評価のための出力形式。

    1. 判断基準:

    各候補レポートについて、次を判定すること。

    臨床的に有意な誤りの件数。
    臨床的に重要でない誤りの件数。

    誤りは次のいずれ
かのカテゴリーに該当する。

    a) 候補報告における虚偽の所見報告。
    b) 参照に存在する所見の欠落。
    c) 所見の解剖学的位置/位置関係の誤同定。
    d) 所見の重症度の誤評価。
    e) 参照にない比較の記載。
    f) 以前の検査からの変化を示す比較の
省略。
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

    [臨床的に有
意な誤り]:
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

REPORT_INSTRUCTIONS = """
Translate the candidate and reference radiology reports from English to Japanese.
Keep JSON keys and integer indices unchanged. Preserve medical meaning exactly,
especially laterality, anatomy, negation, uncertainty, severity, measurements,
temporal comparison, and device positions. Do not correct source errors because
they are part of the evaluation dataset. Preserve anonymization placeholders such
as ___, ____, XXXX, and xxxx exactly. Return only JSON matching the schema.
""".strip()

RESPONSE_INSTRUCTIONS = """
Translate the COMPLETE GREEN evaluation response from English to Japanese.
Every English sentence, phrase, error-category name, finding description, and
section heading must be translated. Do not stop after translating only the
explanation or headings. Do not leave English prose anywhere in the response,
except unavoidable medical abbreviations, anonymization tokens, and isolated
letters used as category labels such as (a) through (f).

Keep JSON keys and integer indices unchanged. Preserve all error counts,
category letters, clinical meaning, laterality, negation, measurements,
comparisons, line breaks, semicolons, and anonymization placeholders exactly.
Do not correct errors in the source.

Translate headings as follows when present:
[Explanation] -> [説明]
[Clinically Significant Errors] -> [臨床的に有意な誤り]
[Clinically Insignificant Errors] -> [臨床的に重要でない誤り]
[Matched Findings] -> [一致する所見]

Translate recurring category labels consistently:
False report of a finding in the candidate -> 候補レポートにおける虚偽の所見報告
Missing a finding present in the reference -> 参照レポートに存在する所見の欠落
Misidentification of a finding's anatomic location/position -> 所見の解剖学的位置／位置関係の誤同定
Misassessment of the severity of a finding -> 所見の重症度の誤評価
Mentioning a comparison that isn't in the reference -> 参照レポートにない比較の記載
Omitting a comparison detailing a change from a prior study -> 前回検査からの変化を示す比較の省略

Return only JSON matching the schema.
""".strip()

REPORT_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "index": {"type": "integer"},
        "candidate": {"type": "string"},
        "reference": {"type": "string"},
    },
    "required": ["index", "candidate", "reference"],
    "additionalProperties": False,
}

RESPONSE_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "index": {"type": "integer"},
        "response": {"type": "string"},
    },
    "required": ["index", "response"],
    "additionalProperties": False,
}

REPORT_BATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "records": {"type": "array", "items": REPORT_ITEM_SCHEMA}
    },
    "required": ["records"],
    "additionalProperties": False,
}

RESPONSE_BATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "records": {"type": "array", "items": RESPONSE_ITEM_SCHEMA}
    },
    "required": ["records"],
    "additionalProperties": False,
}


ENGLISH_RESPONSE_PHRASES = (
    "Clinically Significant Errors",
    "Clinically Insignificant Errors",
    "Matched Findings",
    "False report of a finding",
    "Missing a finding present in the reference",
    "Misidentification of a finding",
    "Misassessment of the severity",
    "Mentioning a comparison",
    "Omitting a comparison",
    "The candidate report",
    "The reference report",
)


def response_translation_looks_complete(source_text: str, translated_text: str) -> tuple[bool, str]:
    """Heuristically reject partially untranslated or truncated responses."""
    if not translated_text.strip():
        return False, "empty translated response"

    # Required translated headings when their English equivalents exist.
    heading_pairs = (
        ("[Explanation]", "[説明]"),
        ("[Clinically Significant Errors]", "[臨床的に有意な誤り]"),
        ("[Clinically Insignificant Errors]", "[臨床的に重要でない誤り]"),
        ("[Matched Findings]", "[一致する所見]"),
    )
    for english, japanese in heading_pairs:
        if english in source_text and japanese not in translated_text:
            return False, f"missing translated heading: {japanese}"

    for phrase in ENGLISH_RESPONSE_PHRASES:
        if phrase.lower() in translated_text.lower():
            return False, f"untranslated English phrase remains: {phrase}"

    # A Japanese translation should contain Japanese characters.
    japanese_chars = len(re.findall(r"[ぁ-んァ-ヶ一-龠]", translated_text))
    if japanese_chars < 10:
        return False, "too few Japanese characters"

    # Reject suspicious truncation. Japanese can be shorter than English, but
    # responses below 35% of the source length are usually incomplete.
    if len(source_text) >= 200 and len(translated_text) < len(source_text) * 0.35:
        return False, "translated response is suspiciously short"

    # Detect long runs of ordinary English prose while allowing abbreviations
    # such as CT, SVC, PICC, ET, NG, and category letters.
    english_words = re.findall(r"\b[A-Za-z]{3,}\b", translated_text)
    allowed = {
        "CT", "MRI", "PET", "SVC", "PICC", "PICC", "ETT", "CABG",
        "COPD", "NG", "IJ", "AP", "PA", "None",
    }
    ordinary = [w for w in english_words if w.upper() not in allowed]
    if len(ordinary) >= 12:
        return False, f"too much English prose remains ({len(ordinary)} words)"

    return True, ""

_tls = threading.local()


def get_client() -> OpenAI:
    if not hasattr(_tls, "client"):
        _tls.client = OpenAI()
    return _tls.client


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Translate candidate/reference with gpt-5-nano and response with "
            "gpt-5-mini, then build prompt locally."
        )
    )
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--errors", type=Path)
    p.add_argument("--report-model", default="gpt-5-nano")
    p.add_argument("--response-model", default="gpt-5-mini")
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
    for i, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Record {i} is not an object.")
        if not isinstance(record.get("key"), dict):
            raise ValueError(f"Record {i}: key must be an object.")
        for field in ("candidate", "reference"):
            if not isinstance(record["key"].get(field), str):
                raise ValueError(f"Record {i}: key.{field} must be a string.")
        if not isinstance(record.get("response"), str):
            raise ValueError(f"Record {i}: response must be a string.")
    return data


def build_prompt(reference: str, candidate: str) -> str:
    return (
        PROMPT_TEMPLATE
        .replace("__REFERENCE_REPORT__", reference)
        .replace("__CANDIDATE_REPORT__", candidate)
    )


def placeholder_counts(text: str) -> dict[str, int]:
    return {token: text.count(token) for token in ("___", "____", "XXXX", "xxxx")}


def call_structured_batch(
    *,
    model: str,
    instructions: str,
    payload: dict[str, Any],
    schema_name: str,
    schema: dict[str, Any],
    max_retries: int,
) -> list[dict[str, Any]]:
    raw_input = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = get_client().responses.create(
                model=model,
                reasoning={"effort": "minimal"},
                instructions=instructions,
                input=raw_input,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
            parsed = json.loads(response.output_text)
            print(
                f"input={response.usage.input_tokens}, "
                f"output={response.usage.output_tokens}, "
                f"total={response.usage.total_tokens}"
            )
            return parsed["records"]
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(60.0, (2 ** attempt) + random.random()))

    raise RuntimeError(
        f"{schema_name} failed after {max_retries + 1} attempts"
    ) from last_error


def translate_reports(
    indices: list[int],
    source: list[dict[str, Any]],
    model: str,
    max_retries: int,
) -> dict[int, dict[str, str]]:
    source_items = [
        {
            "index": i,
            "candidate": source[i]["key"]["candidate"],
            "reference": source[i]["key"]["reference"],
        }
        for i in indices
    ]

    translated = call_structured_batch(
        model=model,
        instructions=REPORT_INSTRUCTIONS,
        payload={"records": source_items},
        schema_name="green_report_translation_batch",
        schema=REPORT_BATCH_SCHEMA,
        max_retries=max_retries,
    )

    src_map = {item["index"]: item for item in source_items}
    dst_map = {item["index"]: item for item in translated}
    if set(src_map) != set(dst_map):
        raise ValueError("Report translation changed record indices.")

    for i, src in src_map.items():
        dst = dst_map[i]
        for field in ("candidate", "reference"):
            if placeholder_counts(src[field]) != placeholder_counts(dst[field]):
                raise ValueError(f"Placeholder mismatch at record {i}, {field}.")

    return {
        i: {
            "candidate": dst_map[i]["candidate"],
            "reference": dst_map[i]["reference"],
        }
        for i in indices
    }


def translate_responses(
    indices: list[int],
    source: list[dict[str, Any]],
    model: str,
    max_retries: int,
) -> dict[int, str]:
    source_items = [
        {"index": i, "response": source[i]["response"]}
        for i in indices
    ]
    src_map = {item["index"]: item for item in source_items}
    last_error: Exception | None = None

    # Retry not only API errors, but also semantically incomplete translations.
    for attempt in range(max_retries + 1):
        try:
            translated = call_structured_batch(
                model=model,
                instructions=RESPONSE_INSTRUCTIONS,
                payload={"records": source_items},
                schema_name="green_response_translation_batch",
                schema=RESPONSE_BATCH_SCHEMA,
                max_retries=0,
            )

            dst_map = {item["index"]: item for item in translated}
            if set(src_map) != set(dst_map):
                raise ValueError("Response translation changed record indices.")

            for i, src in src_map.items():
                translated_response = dst_map[i]["response"]
                if placeholder_counts(src["response"]) != placeholder_counts(
                    translated_response
                ):
                    raise ValueError(
                        f"Placeholder mismatch at record {i}, response."
                    )

                complete, reason = response_translation_looks_complete(
                    src["response"], translated_response
                )
                if not complete:
                    raise ValueError(
                        f"Incomplete response translation at record {i}: {reason}"
                    )

            return {i: dst_map[i]["response"] for i in indices}

        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            delay = min(60.0, (2 ** attempt) + random.random())
            print(
                f"\nResponse batch beginning at {indices[0]} failed validation "
                f"on attempt {attempt + 1}: {exc}\n"
                f"Retrying in {delay:.1f} seconds...",
                file=sys.stderr,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Response batch beginning at {indices[0]} failed after "
        f"{max_retries + 1} attempts"
    ) from last_error


def translate_dual_model_batch(
    indices: list[int],
    source: list[dict[str, Any]],
    report_model: str,
    response_model: str,
    max_retries: int,
) -> dict[int, dict[str, Any]]:
    # The two translations are independent, so run them concurrently.
    with ThreadPoolExecutor(max_workers=2) as pool:
        report_future = pool.submit(
            translate_reports,
            indices,
            source,
            report_model,
            max_retries,
        )
        response_future = pool.submit(
            translate_responses,
            indices,
            source,
            response_model,
            max_retries,
        )
        reports = report_future.result()
        responses = response_future.result()

    final: dict[int, dict[str, Any]] = {}
    for i in indices:
        candidate = reports[i]["candidate"]
        reference = reports[i]["reference"]
        final[i] = {
            "key": {
                "candidate": candidate,
                "reference": reference,
            },
            "response": responses[i],
            "prompt": build_prompt(reference, candidate),
        }
    return final


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                completed[int(row["index"])] = row["record"]
            except Exception as exc:
                raise ValueError(
                    f"Invalid checkpoint line {line_number}: {exc}"
                ) from exc
    return completed


def chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[i:i + size] for i in range(0, len(values), size)]


def main() -> int:
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2
    if args.batch_size < 1 or args.workers < 1:
        raise ValueError("--batch-size and --workers must be >= 1")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")

    checkpoint = args.checkpoint or args.output.with_name(
        args.output.name + ".checkpoint.jsonl"
    )
    errors = args.errors or args.output.with_name(
        args.output.name + ".errors.jsonl"
    )

    if args.overwrite:
        for path in (args.output, checkpoint, errors):
            if path.exists():
                path.unlink()

    source = load_json(args.input)
    if args.start_index >= len(source):
        raise ValueError(f"--start-index must be smaller than {len(source)}")

    stop = len(source)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        stop = min(len(source), args.start_index + args.limit)

    selected = list(range(args.start_index, stop))
    completed = read_checkpoint(checkpoint)
    pending = [i for i in selected if i not in completed]
    batches = chunks(pending, args.batch_size)

    print(
        f"Selected: {len(selected)}, completed: {len(selected) - len(pending)}, "
        f"pending: {len(pending)}"
    )
    print(
        f"batch-size={args.batch_size}, workers={args.workers}, "
        f"report-model={args.report_model}, response-model={args.response_model}"
    )

    failed: list[list[int]] = []
    with tqdm(
        total=len(selected),
        initial=len(selected) - len(pending),
        unit="record",
    ) as bar:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    translate_dual_model_batch,
                    idxs,
                    source,
                    args.report_model,
                    args.response_model,
                    args.max_retries,
                ): idxs
                for idxs in batches
            }

            for future in as_completed(futures):
                idxs = futures[future]
                try:
                    translated = future.result()
                    rows = []
                    for i in idxs:
                        completed[i] = translated[i]
                        rows.append({"index": i, "record": translated[i]})
                    append_jsonl(checkpoint, rows)
                    bar.update(len(idxs))
                except Exception as exc:
                    failed.append(idxs)
                    append_jsonl(
                        errors,
                        [{
                            "indices": idxs,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }],
                    )
                    print(f"\nFailed batch {idxs}: {exc}", file=sys.stderr)

    if failed:
        print(
            "Some records failed. Rerun the same command to retry unfinished records.",
            file=sys.stderr,
        )
        return 1

    missing = [i for i in selected if i not in completed]
    if missing:
        raise RuntimeError(f"Missing records: {missing[:10]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(
            [completed[i] for i in selected],
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")
    temporary.replace(args.output)

    print(f"Completed: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())