import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["FLAGS_use_cuda"] = "0"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

img = sys.argv[1]
out = sys.argv[2]

from paddleocr import PaddleOCR
# Latin recognition model; angle cls off for printed columns
ocr = PaddleOCR(lang="sv", use_doc_orientation_classify=False,
                use_doc_unwarping=False, use_textline_orientation=False,
                device="cpu", enable_mkldnn=False)
result = ocr.predict(img)
lines = []
for res in result:
    txts = res.get("rec_texts", [])
    for t in txts:
        lines.append(t)
text = "\n".join(lines)
with open(out, "w", encoding="utf-8") as f:
    f.write(text)
print("LINES:", len(lines))
print(text[:1500])
