import sys, json, logging, random, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, torch
from shared.dataset import get_dataloaders, get_train_transforms, get_val_transforms
from shared.engine.trainer import Trainer
from shared.engine.evaluator import Evaluator
from shared.utils.checkpoint import load_checkpoint
from shared.utils.metrics import (compute_confusion_matrix,
    compute_classification_report, compute_per_class_metrics, print_results_table)
from shared.utils.visualization import (plot_training_curves,
    plot_confusion_matrix, plot_per_class_metrics, plot_lr_curve)
from model import get_model
from config import ModelConfig as Cfg

def set_seed(s=Cfg.SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int, default=Cfg.EPOCHS)
    p.add_argument("--resume",    type=str, default=None)
    p.add_argument("--data-root", type=str, default=str(Cfg.DATA_ROOT))
    args = p.parse_args()
    Cfg.make_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(str(Cfg.OUTPUT_DIR / "train.log"))])
    set_seed()
    dr = Path(args.data_root)
    train_j = str(dr/"labels"/"det_20"/"det_train.json")
    train_i = str(dr/"images"/"100k"/"train")
    trn_t, val_t = get_train_transforms(Cfg.IMG_SIZE), get_val_transforms(Cfg.IMG_SIZE)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_j, train_i, trn_t, val_t, Cfg.BATCH_SIZE)
    n_tr = len(train_loader.dataset)
    n_va = len(val_loader.dataset)
    n_te = len(test_loader.dataset)
    logging.info(f"[{Cfg.MODEL_NAME}] train={n_tr:,} val={n_va:,} test={n_te:,}")
    model = get_model()
    if args.resume:
        load_checkpoint(args.resume, model, device=Cfg.DEVICE)
    trainer = Trainer(model, train_loader, val_loader, Cfg,
                      device=Cfg.DEVICE, output_dir=str(Cfg.CHECKPOINT_DIR))
    history = trainer.train(epochs=args.epochs)
    out = Path(Cfg.OUTPUT_DIR)
    with open(out/"training_history.json","w") as f: json.dump(history,f,indent=2)
    plots = out/"plots"
    plot_training_curves(history, str(plots/"training_curves.png"))
    plot_lr_curve(history,        str(plots/"lr_schedule.png"))
    best_ckpt = str(Cfg.CHECKPOINT_DIR/"best.pth")
    load_checkpoint(best_ckpt, model, device=Cfg.DEVICE)
    ev  = Evaluator(model, Cfg.DEVICE)
    res = ev.evaluate(test_loader, Cfg.MODEL_NAME+"-Test")
    rep = compute_classification_report(res["all_preds"], res["all_labels"])
    pcm = compute_per_class_metrics(res["all_preds"], res["all_labels"])
    cm  = compute_confusion_matrix(res["all_preds"], res["all_labels"])
    logging.info(f"\n{rep}")
    print_results_table({Cfg.MODEL_NAME: res})
    plot_confusion_matrix(cm,  save_path=str(plots/"confusion_matrix.png"))
    plot_per_class_metrics(pcm,save_path=str(plots/"per_class_metrics.png"))
    with open(out/"test_results.json","w") as f:
        json.dump({"accuracy":res["accuracy"],"inference_s":res["inference_s"],
                   "per_class":pcm},f,indent=2)

if __name__ == "__main__":
    main()
