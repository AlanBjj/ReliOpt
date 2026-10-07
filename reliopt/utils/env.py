"""Project root and .env loading (standard library only, so it also runs under the system python3)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_env(path=None):
    """Read KEY=VALUE lines from .env into os.environ (existing variables win; a missing file is fine),
    and exempt local addresses from the proxy: some login environments export a global http(s)_proxy,
    and clients that honour it route localhost calls through the proxy, which answers 503."""
    p = Path(path) if path else ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.split(" #", 1)[0].strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    local = "localhost,127.0.0.1,0.0.0.0"
    for k in ("NO_PROXY", "no_proxy"):
        cur = os.environ.get(k, "")
        if local not in cur:
            os.environ[k] = f"{cur},{local}" if cur else local
