"""Download pinned public inputs and verify their SHA-256 before use."""

import hashlib
import urllib.request
from pathlib import Path

from benchmarks.public.common import read_json, write_json


def main():
    sources = read_json("benchmarks/public/protocol.json")["sources"]
    for source in sources:
        path = Path(source["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".download")
            with urllib.request.urlopen(source["url"], timeout=60) as response:
                with temporary.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != source["sha256"]:
                temporary.unlink()
                raise ValueError("Downloaded dataset checksum mismatch")
            temporary.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Existing dataset checksum mismatch")
        print(source["name"], "verified")
    write_json("artifacts/public-benchmark/data/sources.json", sources)


if __name__ == "__main__":
    main()
