import sys, json, warnings, traceback
warnings.filterwarnings("ignore")
from datasets import load_dataset

def probe(hid, cfg=None, split="train"):
    try:
        ds = load_dataset(hid, cfg, split=split, streaming=True)
        it = iter(ds)
        row = next(it)
        cols = list(row.keys())
        return ("OK", cols, {k: (str(v)[:60]) for k, v in row.items()})
    except Exception as e:
        return ("FAIL", f"{type(e).__name__}: {str(e)[:160]}", None)

if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        hid = parts[0].strip()
        cfg = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
        split = parts[2].strip() if len(parts) > 2 and parts[2].strip() else "train"
        st, a, b = probe(hid, cfg, split)
        print(json.dumps({"id": hid, "cfg": cfg, "split": split, "status": st, "info": a}), flush=True)
