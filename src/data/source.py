"""Fetch the immutable, checksum-pinned CORGI files before unpickling."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

REPOSITORY = "https://raw.githubusercontent.com/yizhilll/CORGI-PM"


@dataclass(frozen=True)
class SourceFile:
    name: str
    size: int
    sha256: str
    blob_sha1: str
    url: str


def read_lock(path):
    lock = json.loads(Path(path).read_text(encoding="utf-8"))
    commit = lock["source_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Source must use a full Git commit, not a branch")
    specs = []
    for item in lock["files"]:
        name = item["name"]
        if Path(name).name != name or "/" in name or "\\" in name or not name.endswith(".npy"):
            raise ValueError("Unsafe source filename")
        if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) or not re.fullmatch(r"[0-9a-f]{40}", item["github_blob_sha1"]):
            raise ValueError("Invalid source checksum")
        if type(item["bytes"]) is not int or item["bytes"] <= 0:
            raise ValueError("Invalid source size")
        specs.append(SourceFile(name, item["bytes"], item["sha256"], item["github_blob_sha1"], f"{REPOSITORY}/{commit}/dataset/{name}"))
    if len(specs) != 2 or len({x.name for x in specs}) != 2:
        raise ValueError("Expected the two distinct CORGI source files")
    return lock, specs


def verify_source(path, spec):
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
    if len(raw) != spec.size or sha != spec.sha256 or blob != spec.blob_sha1:
        raise ValueError(f"{spec.name}: source size/checksum mismatch; refusing to load npy")
    return {"name": spec.name, "bytes": len(raw), "sha256": sha, "github_blob_sha1": blob, "url": spec.url}


def fetch_source(raw_dir, spec, *, offline=False, attempts=3):
    directory = Path(raw_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / spec.name
    if target.exists():
        return target, verify_source(target, spec)
    if offline:
        raise FileNotFoundError(f"{target}: missing source in offline mode")
    for attempt in range(attempts):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".p1-download-", dir=directory, delete=False) as stream:
                temporary = Path(stream.name)
                request = Request(spec.url, headers={"User-Agent": "NLP-Course-P1/1.0"})
                with urlopen(request, timeout=30) as response:
                    total = 0
                    while chunk := response.read(128 * 1024):
                        total += len(chunk)
                        if total > spec.size:
                            raise ValueError(f"{spec.name}: download exceeds locked size")
                        stream.write(chunk)
            result = verify_source(temporary, spec)
            # Only the newly created temporary file is replaced/removed here.
            if target.exists():
                verify_source(target, spec)
            else:
                temporary.rename(target)
            return target, result
        except (URLError, TimeoutError, ConnectionError):
            if attempt + 1 == attempts:
                raise
            time.sleep(attempt + 1)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise RuntimeError("Source download did not complete")


def load_sources(raw_dir, lock_path, *, offline=False):
    lock, specs = read_lock(lock_path)
    verified = []
    paths = {}
    for spec in specs:
        path, evidence = fetch_source(raw_dir, spec, offline=offline)
        paths[spec.name] = path
        verified.append(evidence)
    # numpy object arrays use pickle: both immutable files are verified first.
    import numpy as np
    corpora = {name: np.load(path, allow_pickle=True).item() for name, path in paths.items()}
    for name, corpus in corpora.items():
        if not isinstance(corpus, dict) or set(corpus) != {"train", "valid", "test"}:
            raise ValueError(f"{name}: invalid source split structure")
    return lock, corpora, verified
