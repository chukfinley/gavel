import sys, json, warnings
warnings.filterwarnings("ignore")
from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import HfApi

def rows_total(hid, cfg, split):
    try:
        from datasets import load_dataset_builder
        b = load_dataset_builder(hid, cfg)
        si = b.info.splits
        if si and split in si:
            return si[split].num_examples
        if si:
            return {k: v.num_examples for k, v in si.items()}
    except Exception:
        pass
    return None

for line in sys.stdin:
    line = line.strip()
    if not line or line.startswith("#"): continue
    p = line.split("|")
    hid = p[0].strip(); cfg = p[1].strip() or None if len(p)>1 else None
    split = (p[2].strip() if len(p)>2 and p[2].strip() else "train")
    try:
        d = load_dataset(hid, cfg, split=f"{split}[:5]")
        cols = list(d.features.keys())
        tot = rows_total(hid, cfg, split)
        print(json.dumps({"id":hid,"cfg":cfg,"split":split,"status":"OK","cols":cols,"total":tot,
                          "sample":{k:str(d[0][k])[:120] for k in cols}}), flush=True)
    except Exception as e:
        print(json.dumps({"id":hid,"cfg":cfg,"split":split,"status":"FAIL","err":f"{type(e).__name__}: {str(e)[:200]}"}), flush=True)
