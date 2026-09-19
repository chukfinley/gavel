import json, urllib.request, sys, time
def s(term, limit=20):
    url=f"https://huggingface.co/api/models?search={urllib.parse.quote(term)}&limit={limit}&sort=downloads&direction=-1"
    try:
        d=json.load(urllib.request.urlopen(url, timeout=30))
    except Exception as e:
        print("ERR", term, e); return
    print(f"=== {term} ===")
    for m in d: print(" ", m['id'], m.get('downloads'))
import urllib.parse
for t in sys.argv[1:]:
    s(t); time.sleep(0.5)
