import sys, json, logging, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.dataset import get_dataloaders, get_val_transforms
from shared.engine.evaluator import Evaluator
from shared.utils.checkpoint import load_checkpoint
from shared.utils.metrics import (compute_confusion_matrix,
    compute_classification_report, compute_per_class_metrics, print_results_table)
from shared.utils.visualization import plot_confusion_matrix, plot_per_class_metrics
from model import get_model
from config import ModelConfig as Cfg

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",      default=str(Cfg.CHECKPOINT_DIR/"best.pth"))
    p.add_argument("--data-root", default=str(Cfg.DATA_ROOT))
    p.add_argument("--output",    default=str(Cfg.OUTPUT_DIR/"eval"))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    Cfg.make_dirs()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    dr  = Path(args.data_root)
    model = get_model()
    load_checkpoint(args.ckpt, model, device=Cfg.DEVICE)
    ev  = Evaluator(model, Cfg.DEVICE)
    val_t = get_val_transforms(Cfg.IMG_SIZE)
    _, _, test_loader = get_dataloaders(
        str(dr/"labels"/"det_20"/"det_train.json"),
        str(dr/"images"/"100k"/"train"),
        val_t, val_t, Cfg.BATCH_SIZE)
    res = ev.evaluate(test_loader, Cfg.MODEL_NAME)
    rep = compute_classification_report(res["all_preds"], res["all_labels"])
    pcm = compute_per_class_metrics(res["all_preds"], res["all_labels"])
    cm  = compute_confusion_matrix(res["all_preds"], res["all_labels"])
    logging.info(f"\n{rep}")
    print_results_table({Cfg.MODEL_NAME: res})
    slug = Cfg.MODEL_NAME
    plot_confusion_matrix(cm,   save_path=str(out/f"cm_{slug}.png"))
    plot_per_class_metrics(pcm, save_path=str(out/f"pcm_{slug}.png"))
    with open(out/"eval_results.json","w") as f:
        json.dump({"accuracy":res["accuracy"],"inference_s":res["inference_s"],
                   "per_class":pcm},f,indent=2)

if __name__ == "__main__":
    main()
