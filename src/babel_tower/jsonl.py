"""Read and write lists of record dataclasses as JSONL."""

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


def read[T](path: Path, cls: type[T]) -> list[T]:
    with path.open(encoding="utf-8") as f:
        return [cls(**json.loads(line)) for line in f]


def write(path: Path, rows: Iterable[DataclassInstance]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(dumps(asdict(row)) + "\n")


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)
