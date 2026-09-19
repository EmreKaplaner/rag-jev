"""Fetch the exact public sources and reconstruct the committed pilot protocol inputs."""

import hashlib
import shutil
import subprocess
import urllib.request

from benchmarks.ecosystem.common import ROOT, fingerprint_files
from benchmarks.ecosystem.prepare import beir, crag
from benchmarks.public.common import read_json, write_json

REPOS = {
    "flashrag": "https://github.com/RUC-NLPIR/FlashRAG.git",
    "bergen": "https://github.com/naver/bergen.git",
    "crag": "https://github.com/facebookresearch/CRAG.git",
    "ragchecker": "https://github.com/amazon-science/RAGChecker.git",
}


def main():
    p = read_json("benchmarks/ecosystem/protocol.json")
    assert fingerprint_files(p["code"]) == p["code"]
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "protocol.json").exists():
        assert read_json(ROOT / "protocol.json") == p
    for name, sha in p["upstream"].items():
        path = ROOT / "upstream" / name
        if not path.exists():
            subprocess.run(
                ["git", "clone", "--no-checkout", "--filter=blob:none", REPOS[name], str(path)],
                check=True,
            )
            subprocess.run(["git", "-C", str(path), "checkout", sha], check=True)
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        assert actual == sha, f"{name} upstream mismatch"
    for name, source in p["sources"].items():
        path = ROOT / "data" / ("crag.bz2" if name == "crag" else name + ".zip")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            with (
                urllib.request.urlopen(source["url"], timeout=120) as response,
                temporary.open("wb") as out,
            ):
                shutil.copyfileobj(response, out)
            temporary.replace(path)
        with path.open("rb") as f:
            assert hashlib.file_digest(f, "sha256").hexdigest() == source["sha256"]
        write_json(path.parent / f"{name}.source.json", source)
    for name in ["fiqa", "nfcorpus", "scifact", "crag"]:
        result = crag() if name == "crag" else beir(name)
        assert result == p["datasets"][name], f"{name} input reproduction mismatch"
    write_json(ROOT / "protocol.json", p)
    print("Exact pilot sources and sampled cases reconstructed. No paid calls.")


if __name__ == "__main__":
    main()
