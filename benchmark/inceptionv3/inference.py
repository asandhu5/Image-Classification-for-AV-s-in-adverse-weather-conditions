import sys, argparse, json, logging, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, torch.nn.functional as F
from PIL import Image
from shared.utils.checkpoint import load_checkpoint
from shared.dataset.transforms import get_val_transforms
from shared.config import SharedConfig as C
from model import get_model
from config import ModelConfig as Cfg

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source",  required=True)
    p.add_argument("--ckpt",    default=str(Cfg.CHECKPOINT_DIR/"best.pth"))
    p.add_argument("--device",  default=Cfg.DEVICE)
    return p.parse_args()

@torch.inference_mode()
def classify(model, transform, img_path, device):
    img = Image.open(img_path).convert("RGB")
    x   = transform(img).unsqueeze(0).to(device)
    t0  = time.perf_counter()
    out = model(x)
    elapsed = time.perf_counter() - t0
    logits  = out[0] if isinstance(out, tuple) else out
    probs   = F.softmax(logits, dim=-1)[0]
    v5, i5  = probs.topk(5)
    return {
        "file":        str(img_path),
        "class_idx":   int(i5[0]),
        "class_name":  C.IDX2CLASS[int(i5[0])],
        "confidence":  round(float(v5[0]),4),
        "top5":        [{"class":C.IDX2CLASS[int(i5[k])],
                         "prob": round(float(v5[k]),4)} for k in range(5)],
        "inference_s": round(elapsed,4),
    }

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    model = get_model()
    load_checkpoint(args.ckpt, model, device=args.device)
    model.eval()
    transform = get_val_transforms(Cfg.IMG_SIZE)
    src  = Path(args.source)
    imgs = list(src.glob("*.jpg"))+list(src.glob("*.png")) if src.is_dir() else [src]
    for img_path in imgs:
        print(json.dumps(classify(model, transform, img_path, args.device), indent=2))

if __name__ == "__main__":
    main()
