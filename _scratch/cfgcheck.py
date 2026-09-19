import sys, warnings
warnings.filterwarnings("ignore")
from transformers import AutoConfig
ids = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
for i in ids:
    try:
        c = AutoConfig.from_pretrained(i, trust_remote_code=True)
        arch = getattr(c,'architectures',None)
        print(f"{i:60s} | {c.model_type:22s} | mpe={getattr(c,'max_position_embeddings',None)} | L={getattr(c,'num_hidden_layers',None)} | H={getattr(c,'hidden_size',None)} | V={getattr(c,'vocab_size',None)} | arch={arch} | rope={getattr(c,'rope_scaling',None)}")
    except Exception as e:
        print(f"{i:60s} | ERROR {type(e).__name__}: {str(e)[:150]}")
