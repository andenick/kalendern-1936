import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["USE_TORCH"] = "1"

img = sys.argv[1]
out = sys.argv[2]

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(pretrained=True)
doc = DocumentFile.from_images(img)
result = model(doc)
text = result.render()
with open(out, "w", encoding="utf-8") as f:
    f.write(text)
print("CHARS:", len(text))
print(text[:1500])
