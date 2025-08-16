import json, os, tempfile, time

SCHEMA_VERSION = 1

def _atomic_write(path: str, data: str, encoding="utf-8"):
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on same filesystem
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

def save_project(path: str, payload: dict):
    payload = dict(payload or {})
    payload["schema_version"] = SCHEMA_VERSION
    payload["saved_at"] = int(time.time())
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))

def load_project(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj or {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
