import sys, warnings, torch
warnings.filterwarnings("ignore")
from transformers import AutoModelForSequenceClassification, AutoTokenizer
for i in [l.strip() for l in open(sys.argv[1]) if l.strip()]:
    try:
        m = AutoModelForSequenceClassification.from_pretrained(i, num_labels=3, trust_remote_code=True, dtype=torch.float32)
        n = sum(p.numel() for p in m.parameters())/1e6
        cls = type(m).__name__
        # quick forward at 16k if it fits cheaply
        print(f"OK   {i:58s} {n:8.1f}M  {cls}")
        del m
    except Exception as e:
        print(f"FAIL {i:58s} {type(e).__name__}: {str(e)[:180]}")
