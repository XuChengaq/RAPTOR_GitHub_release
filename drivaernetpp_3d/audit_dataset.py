from __future__ import annotations

import argparse
import collections
import io
import json
import struct
import zipfile
from pathlib import Path

import numpy as np


class SliceReader(io.RawIOBase):
    """Seekable view over a stored nested ZIP without extracting the outer archive."""

    def __init__(self, path: Path, start: int, length: int):
        self._handle = path.open("rb")
        self._start = start
        self._length = length
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._position + offset
        elif whence == io.SEEK_END:
            position = self._length + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if position < 0:
            raise ValueError("negative seek")
        self._position = min(position, self._length)
        return self._position

    def readinto(self, buffer: bytearray) -> int:
        remaining = self._length - self._position
        if remaining <= 0:
            return 0
        size = min(len(buffer), remaining)
        self._handle.seek(self._start + self._position)
        data = self._handle.read(size)
        buffer[: len(data)] = data
        self._position += len(data)
        return len(data)

    def close(self) -> None:
        self._handle.close()
        super().close()


def nested_data_offset(path: Path, info: zipfile.ZipInfo) -> int:
    with path.open("rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
    signature, *fields = struct.unpack("<IHHHHHIIIHH", header)
    if signature != 0x04034B50:
        raise ValueError("invalid local ZIP header")
    name_length, extra_length = fields[-2:]
    return info.header_offset + 30 + name_length + extra_length


def array_metadata(array: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "min": float(array.min()) if array.size else None,
        "max": float(array.max()) if array.size else None,
    }


def audit(archive_path: Path, sample_per_family: int = 2) -> dict[str, object]:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with zipfile.ZipFile(archive_path) as outer:
        bad_outer = outer.testzip()
        nested = next(item for item in outer.infolist() if item.filename.lower().endswith(".zip"))
        if nested.compress_type != zipfile.ZIP_STORED:
            raise RuntimeError("nested ZIP is compressed; safe direct slicing is unavailable")
        start = nested_data_offset(archive_path, nested)

    family_counts: collections.Counter[str] = collections.Counter()
    samples: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    with SliceReader(archive_path, start, nested.file_size) as raw:
        with zipfile.ZipFile(io.BufferedReader(raw)) as inner:
            vehicle_files = [
                item for item in inner.infolist()
                if not item.is_dir() and item.filename.startswith("DrivAerNet/")
                and item.filename.lower().endswith(".npz")
            ]
            for item in vehicle_files:
                parts = item.filename.split("/")
                family = parts[1] if len(parts) > 2 else "unknown"
                family_counts[family] += 1
                if len(samples[family]) >= sample_per_family:
                    continue
                with inner.open(item) as handle:
                    payload = np.load(io.BytesIO(handle.read()), allow_pickle=False)
                    arrays = {key: array_metadata(payload[key]) for key in payload.files}
                required = {"nodes_list", "elems_list", "features_list", "elem_features_list"}
                samples[family].append({
                    "path": item.filename,
                    "required_fields_present": required.issubset(arrays),
                    "arrays": arrays,
                })
            bad_inner = inner.testzip()

    return {
        "source_archive": str(archive_path.resolve()),
        "outer_integrity_error": bad_outer,
        "nested_archive": nested.filename,
        "nested_integrity_error": bad_inner,
        "vehicle_file_count": sum(family_counts.values()),
        "vehicle_family_counts": dict(sorted(family_counts.items())),
        "sample_schema": dict(samples),
        "target_scope": "single-channel vehicle surface node/element field",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("dataset_audit.json"))
    parser.add_argument("--sample-per-family", type=int, default=2)
    args = parser.parse_args()
    report = audit(args.archive, args.sample_per_family)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
