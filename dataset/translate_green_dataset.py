#!/usr/bin/env python3
"""
Translate the GREEN dataset JSON from English to Japanese while preserving
the JSON keys: key, candidate, reference, response, prompt.

Features
--------
- Translates one record per API request.
- Preserves the five specified JSON keys in English.
- Uses Structured Outputs so each translated record remains valid JSON.
- Writes a JSONL checkpoint after every successful record.
- Can resume safely after interruption.
- Retries transient API/JSON errors with exponential backoff.
- Supports --limit for a small test run.

Install
-------
pip install -U openai tqdm

Set API key
-----------
export OPENAI_API_KEY="sk-..."

Example
-------
python translate_green_dataset.py \
  --input test.json \
  --output test_ja.json \
  --checkpoint test_ja.checkpoint.jsonl \
  --model gpt-5-mini \
  --limit 10

After checking the first 10 records, run the whole dataset without --limit.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


TRANSLATION_INSTRUCTIONS = r"""
You are a professional Japanese medical translator specializing in radiology.

Translate every English string value in the supplied GREEN dataset record into
natural, accurate Japanese.

Mandatory rules:
1. Preserve the JSON structure exactly.
2. Keep these JSON key names in English exactly as written:
   "key", "candidate", "reference", "response", "prompt".
3. Do not add, remove, rename, or reorder fields.
4. Translate all string values, including the radiology reports, explanations,
   error-category descriptions, instructions, and section headings.
5. Preserve placeholders and anonymization tokens exactly, including:
   ___, ____, XXXX, xxxx, dates with underscores, and similar masked text.
6. Preserve all numbers, measurements, laterality, anatomy, negation,
   uncertainty, temporal comparison, severity, and clinical meaning.
7. Do not correct factual or medical errors in the source. Translate them
   faithfully, even when the candidate report is medically wrong.
8. Do not summarize, omit, or expand the text.
9. Preserve line breaks and list structure inside strings as closely as possible.
10. Use standard Japanese radiology terminology.
11. Return only the translated JSON object matching the required schema.
""".strip()


RECORD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
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
        "prompt": {"type": "string"},
    },
    "required": ["key", "response", "prompt"],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate the GREEN JSON dataset into Japanese."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSON file")
    parser.add_argument("--output", required=True, type=Path, help="Final output JSON file")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint JSONL file. Default: <output>.checkpoint.jsonl",
    )
    parser.add_argument(
        "--errors",
        type=Path,
        default=None,
        help="Error log JSONL file. Default: <output>.errors.jsonl",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="OpenAI model name (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Translate only the first N records for testing",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start from this zero-based record index",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Maximum retries per record (default: 6)",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Seconds to wait after each successful request",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing checkpoint and start over",
    )
    return parser.parse_args()


def derive_sidecar_path(output: Path, suffix: str) -> Path:
    return output.with_name(output.name + suffix)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("The top-level JSON value must be a list.")

    for i, record in enumerate(data):
        validate_source_record(record, i)

    return data


def validate_source_record(record: Any, index: int) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"Record {index}: expected an object.")

    expected_top = {"key", "response", "prompt"}
    if set(record.keys()) != expected_top:
        raise ValueError(
            f"Record {index}: expected top-level keys {sorted(expected_top)}, "
            f"found {sorted(record.keys())}."
        )

    key_obj = record.get("key")
    if not isinstance(key_obj, dict):
        raise ValueError(f"Record {index}: 'key' must be an object.")

    expected_inner = {"candidate", "reference"}
    if set(key_obj.keys()) != expected_inner:
        raise ValueError(
            f"Record {index}: expected keys under 'key' "
            f"{sorted(expected_inner)}, found {sorted(key_obj.keys())}."
        )

    for field_name, value in (
        ("key.candidate", key_obj["candidate"]),
        ("key.reference", key_obj["reference"]),
        ("response", record["response"]),
        ("prompt", record["prompt"]),
    ):
        if not isinstance(value, str):
            raise ValueError(f"Record {index}: '{field_name}' must be a string.")


def validate_translated_record(
    source: dict[str, Any], translated: dict[str, Any], index: int
) -> None:
    validate_source_record(translated, index)

    if list(translated.keys()) != list(source.keys()):
        raise ValueError(f"Record {index}: top-level key order changed.")

    if list(translated["key"].keys()) != list(source["key"].keys()):
        raise ValueError(f"Record {index}: key order inside 'key' changed.")

    # Important placeholders should not disappear during translation.
    for field_path, src_text, dst_text in (
        (
            "key.candidate",
            source["key"]["candidate"],
            translated["key"]["candidate"],
        ),
        (
            "key.reference",
            source["key"]["reference"],
            translated["key"]["reference"],
        ),
        ("response", source["response"], translated["response"]),
        ("prompt", source["prompt"], translated["prompt"]),
    ):
        for token in ("___", "____", "XXXX", "xxxx"):
            if src_text.count(token) != dst_text.count(token):
                raise ValueError(
                    f"Record {index}: placeholder count changed in {field_path}: "
                    f"{token!r} {src_text.count(token)} -> {dst_text.count(token)}"
                )


def translate_record(
    client: OpenAI,
    record: dict[str, Any],
    model: str,
    index: int,
    max_retries: int,
) -> dict[str, Any]:
    payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                instructions=TRANSLATION_INSTRUCTIONS,
                input=payload,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "green_translation_record",
                        "schema": RECORD_SCHEMA,
                        "strict": True,
                    }
                },
            )

            translated = json.loads(response.output_text)
            validate_translated_record(record, translated, index)
            return translated

        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break

            delay = min(60.0, (2**attempt) + random.uniform(0.0, 1.0))
            print(
                f"\nRecord {index}: attempt {attempt + 1} failed: {exc}\n"
                f"Retrying in {delay:.1f} seconds...",
                file=sys.stderr,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Record {index} failed after {max_retries + 1} attempts"
    ) from last_error


def read_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}

    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                index = item["index"]
                record = item["record"]

                if not isinstance(index, int):
                    raise TypeError("'index' must be an integer")

                validate_source_record(record, index)
                completed[index] = record
            except Exception as exc:
                raise ValueError(
                    f"Invalid checkpoint entry at line {line_number}: {exc}"
                ) from exc

    return completed


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_final_json(
    output_path: Path,
    source_data: list[dict[str, Any]],
    completed: dict[int, dict[str, Any]],
    selected_indices: list[int],
) -> None:
    missing = [i for i in selected_indices if i not in completed]
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise RuntimeError(
            f"Cannot create final JSON: {len(missing)} records are missing "
            f"(first indices: {preview})."
        )

    # When only a range or limit is requested, output only that selected subset.
    translated_data = [completed[i] for i in selected_indices]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    temporary_path.replace(output_path)


def main() -> int:
    args = parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Example:\n"
            '  export OPENAI_API_KEY="sk-..."',
            file=sys.stderr,
        )
        return 2

    checkpoint_path = args.checkpoint or derive_sidecar_path(
        args.output, ".checkpoint.jsonl"
    )
    errors_path = args.errors or derive_sidecar_path(args.output, ".errors.jsonl")

    if args.overwrite:
        for path in (checkpoint_path, errors_path, args.output):
            if path.exists():
                path.unlink()

    source_data = load_dataset(args.input)
    total_records = len(source_data)

    if args.start_index < 0 or args.start_index >= total_records:
        raise ValueError(
            f"--start-index must be between 0 and {total_records - 1}."
        )

    end_index = total_records
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be a positive integer.")
        end_index = min(total_records, args.start_index + args.limit)

    selected_indices = list(range(args.start_index, end_index))
    completed = read_checkpoint(checkpoint_path)

    # Ignore checkpoint entries outside the current selected range.
    already_done = sum(i in completed for i in selected_indices)

    print(f"Input records:       {total_records}")
    print(f"Selected range:      {args.start_index}..{end_index - 1}")
    print(f"Already translated: {already_done}")
    print(f"Model:               {args.model}")
    print(f"Checkpoint:          {checkpoint_path}")
    print(f"Output:              {args.output}")

    client = OpenAI()

    failed_indices: list[int] = []

    with tqdm(total=len(selected_indices), initial=already_done, unit="record") as bar:
        for index in selected_indices:
            if index in completed:
                continue

            try:
                translated = translate_record(
                    client=client,
                    record=source_data[index],
                    model=args.model,
                    index=index,
                    max_retries=args.max_retries,
                )

                append_jsonl(
                    checkpoint_path,
                    {"index": index, "record": translated},
                )
                completed[index] = translated
                bar.update(1)

                if args.request_delay > 0:
                    time.sleep(args.request_delay)

            except Exception as exc:
                failed_indices.append(index)
                append_jsonl(
                    errors_path,
                    {
                        "index": index,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                print(
                    f"\nRecord {index} failed and was logged to {errors_path}: {exc}",
                    file=sys.stderr,
                )

    if failed_indices:
        print(
            "\nTranslation finished with errors. "
            f"Failed record indices: {failed_indices}\n"
            "Run the same command again to retry only unfinished records.",
            file=sys.stderr,
        )
        return 1

    write_final_json(
        output_path=args.output,
        source_data=source_data,
        completed=completed,
        selected_indices=selected_indices,
    )

    print(f"\nCompleted successfully: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
