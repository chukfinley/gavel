import warnings, torch
warnings.filterwarnings("ignore")
from transformers import AutoModelForSequenceClassification, AutoTokenizer

mid = "llm-semantic-router/Vela-1.0-Encoder-307M"
tok = AutoTokenizer.from_pretrained(mid)
m = AutoModelForSequenceClassification.from_pretrained(
    mid, num_labels=3, trust_remote_code=True, dtype=torch.float32).eval()
print("params(M):", sum(p.numel() for p in m.parameters()) / 1e6)
print("class:", type(m).__name__)
print("mpe:", m.config.max_position_embeddings)
print("tokenizer model_max_length:", tok.model_max_length)

for L in (4096, 16384):
    ids = torch.randint(5, 250000, (1, L))
    ids[0, 0] = tok.cls_token_id
    ids[0, -1] = tok.sep_token_id
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=torch.ones_like(ids))
    print(f"FWD len={L:6d} logits={tuple(out.logits.shape)} "
          f"finite={torch.isfinite(out.logits).all().item()} "
          f"vals={[round(v,3) for v in out.logits[0].tolist()]}")

enc = tok("premise text here", "hypothesis text here", return_tensors="pt")
print("PAIR keys:", list(enc.keys()), "shape:", tuple(enc.input_ids.shape))
