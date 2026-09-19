import warnings, torch
warnings.filterwarnings("ignore")
from transformers import AutoConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES as M
for mt in ["modernbert","longformer","led","big_bird","nystromformer","xlm-roberta","qwen3","eurobert","new"]:
    print(f"  SeqCls mapping[{mt}] = {M.get(mt)}")
from transformers import AutoModelForSequenceClassification as A
for mid in ["answerdotai/ModernBERT-base"]:
    m=A.from_pretrained(mid,num_labels=3); print("OK", mid, round(sum(p.numel() for p in m.parameters())/1e6,1),"M", type(m).__name__)
