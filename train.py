#!/usr/bin/env python3
"""
Entrenamiento NAS desde cero para segmentacion semantica.

El script espera mascaras PNG con IDs de clase:
  0 = fondo
  1..N-1 = objetos/clases semanticas

Por defecto no carga pesos preentrenados ni warm start. Optuna busca
arquitectura, encoder, loss, batch size e hiperparametros de entrenamiento.
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import optuna
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SEARCH_SPACE = BASE_DIR / "configs" / "search_space.yaml"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

ARCH_REGISTRY = {
    "Unet": smp.Unet,
    "UnetPlusPlus": smp.UnetPlusPlus,
    "FPN": smp.FPN,
    "DeepLabV3Plus": smp.DeepLabV3Plus,
    "Linknet": smp.Linknet,
}


def read_yaml(path):
    if not path or not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def split_csv(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_encoder_weights(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"none", "null", "false", "0"}:
        return None
    return value


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_class_config(path, num_classes):
    names = ["background"] + [f"class_{i}" for i in range(1, num_classes)]
    color_to_id = {}
    if not path:
        return names, color_to_id

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"No existe class config: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    classes = cfg.get("classes", [])
    for item in classes:
        class_id = int(item["id"])
        if 0 <= class_id < num_classes:
            names[class_id] = str(item.get("name", names[class_id]))
            if "color" in item:
                color = tuple(int(v) for v in item["color"])
                if len(color) == 3:
                    color_to_id[color] = class_id
    return names, color_to_id


def default_palette(num_classes):
    fixed = [
        (0, 0, 0),
        (0, 255, 0),
        (255, 128, 0),
        (0, 128, 255),
        (255, 0, 128),
        (180, 255, 0),
        (160, 80, 255),
        (255, 220, 0),
        (0, 220, 220),
    ]
    palette = []
    for idx in range(num_classes):
        if idx < len(fixed):
            palette.append(fixed[idx])
        else:
            palette.append(((37 * idx) % 256, (91 * idx) % 256, (173 * idx) % 256))
    return np.asarray(palette, dtype=np.uint8)


def build_train_transform(img_size, aug_level):
    transforms = [A.Resize(img_size, img_size)]
    if aug_level in {"light", "medium", "strong"}:
        transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.12,
                rotate_limit=35,
                p=0.45,
                border_mode=cv2.BORDER_REFLECT_101,
            ),
            A.RandomBrightnessContrast(p=0.35),
        ])
    if aug_level in {"medium", "strong"}:
        transforms.extend([
            A.OneOf([
                A.GridDistortion(p=1.0, num_steps=5, distort_limit=0.25),
                A.ElasticTransform(p=1.0, alpha=80, sigma=6),
            ], p=0.2),
            A.RandomGamma(gamma_limit=(85, 120), p=0.2),
            A.GaussNoise(p=0.15),
        ])
    if aug_level == "strong":
        transforms.extend([
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.25),
            A.CLAHE(p=0.2),
        ])
    transforms.extend([
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
    return A.Compose(transforms)


def build_val_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform, num_classes, ignore_index, color_to_id=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.color_to_id = color_to_id or {}
        self.images = sorted(
            path for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        )

    def __len__(self):
        return len(self.images)

    def _read_mask(self, mask_path, shape_hw):
        if not mask_path.exists():
            return np.zeros(shape_hw, dtype=np.int64)

        raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            return np.zeros(shape_hw, dtype=np.int64)

        if raw.ndim == 3:
            rgb = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB)
            mask = np.zeros(rgb.shape[:2], dtype=np.int64)
            if not self.color_to_id:
                raise ValueError(
                    f"La mascara {mask_path} es RGB. Usa mascaras con IDs o pasa --class-config con colores."
                )
            matched = np.zeros(rgb.shape[:2], dtype=bool)
            for color, class_id in self.color_to_id.items():
                color_arr = np.asarray(color, dtype=np.uint8)
                pixels = np.all(rgb == color_arr, axis=-1)
                mask[pixels] = class_id
                matched |= pixels
            if np.any(~matched):
                unknown = rgb[~matched][0].tolist()
                raise ValueError(f"Color no definido en {mask_path}: {unknown}")
            return mask

        mask = raw.astype(np.int64)
        if self.num_classes == 2:
            if self.ignore_index >= 0:
                void = mask == self.ignore_index
                mask = (mask > 0).astype(np.int64)
                mask[void] = self.ignore_index
            elif mask.max(initial=0) > 1:
                mask = (mask > 0).astype(np.int64)
        else:
            valid = np.ones(mask.shape, dtype=bool)
            if self.ignore_index >= 0:
                valid &= mask != self.ignore_index
            invalid = valid & ((mask < 0) | (mask >= self.num_classes))
            if np.any(invalid):
                bad_value = int(mask[invalid][0])
                raise ValueError(
                    f"Valor de clase invalido en {mask_path}: {bad_value}. "
                    f"num_classes={self.num_classes}, ignore_index={self.ignore_index}"
                )
        return mask

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"No se pudo leer la imagen: {image_path}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask_path = self.mask_dir / f"{image_path.stem}.png"
        mask = self._read_mask(mask_path, image.shape[:2])

        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented["image"]
        mask_tensor = augmented["mask"].long()
        return image_tensor, mask_tensor, str(image_path)


class SoftDiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=-1, include_background=False, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits, target):
        valid = torch.ones_like(target, dtype=torch.bool)
        safe_target = target
        if self.ignore_index >= 0:
            valid = target != self.ignore_index
            safe_target = target.clone()
            safe_target[~valid] = 0

        prob = torch.softmax(logits, dim=1)
        one_hot = torch.nn.functional.one_hot(safe_target, self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1)
        prob = prob * valid
        one_hot = one_hot * valid

        start_class = 0 if self.include_background else 1
        if start_class >= self.num_classes:
            start_class = 0
        prob = prob[:, start_class:]
        one_hot = one_hot[:, start_class:]

        dims = (0, 2, 3)
        intersection = (prob * one_hot).sum(dims)
        denominator = prob.sum(dims) + one_hot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, dice_weight=0.5, ignore_index=-1, include_background=False):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index if ignore_index >= 0 else -100)
        self.dice = SoftDiceLoss(num_classes, ignore_index, include_background)

    def forward(self, logits, target):
        return self.dice_weight * self.dice(logits, target) + (1.0 - self.dice_weight) * self.ce(logits, target)


class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes, dice_weight=0.5, gamma=2.0, ignore_index=-1, include_background=False):
        super().__init__()
        self.dice_weight = dice_weight
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.dice = SoftDiceLoss(num_classes, ignore_index, include_background)

    def forward(self, logits, target):
        ce = torch.nn.functional.cross_entropy(
            logits,
            target,
            ignore_index=self.ignore_index if self.ignore_index >= 0 else -100,
            reduction="none",
        )
        if self.ignore_index >= 0:
            valid = target != self.ignore_index
            ce = ce[valid]
        if ce.numel() == 0:
            focal = logits.sum() * 0.0
        else:
            pt = torch.exp(-ce)
            focal = ((1.0 - pt) ** self.gamma * ce).mean()
        return self.dice_weight * self.dice(logits, target) + (1.0 - self.dice_weight) * focal


class TverskyCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, tversky_weight=0.7, alpha=0.35, beta=0.65, ignore_index=-1):
        super().__init__()
        self.num_classes = num_classes
        self.tversky_weight = tversky_weight
        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index if ignore_index >= 0 else -100)

    def forward(self, logits, target):
        valid = torch.ones_like(target, dtype=torch.bool)
        safe_target = target
        if self.ignore_index >= 0:
            valid = target != self.ignore_index
            safe_target = target.clone()
            safe_target[~valid] = 0

        prob = torch.softmax(logits, dim=1)
        one_hot = torch.nn.functional.one_hot(safe_target, self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1)
        prob = prob * valid
        one_hot = one_hot * valid

        prob = prob[:, 1:] if self.num_classes > 1 else prob
        one_hot = one_hot[:, 1:] if self.num_classes > 1 else one_hot

        dims = (0, 2, 3)
        tp = (prob * one_hot).sum(dims)
        fp = (prob * (1.0 - one_hot)).sum(dims)
        fn = ((1.0 - prob) * one_hot).sum(dims)
        tversky = (tp + 1.0) / (tp + self.alpha * fp + self.beta * fn + 1.0)
        tversky_loss = 1.0 - tversky.mean()
        return self.tversky_weight * tversky_loss + (1.0 - self.tversky_weight) * self.ce(logits, target)


def make_loss(name, num_classes, dice_weight, ignore_index, include_background):
    if name == "dice_ce":
        return DiceCrossEntropyLoss(num_classes, dice_weight, ignore_index, include_background)
    if name == "focal_dice":
        return FocalDiceLoss(num_classes, dice_weight, ignore_index=ignore_index, include_background=include_background)
    if name == "tversky_ce":
        return TverskyCrossEntropyLoss(num_classes, ignore_index=ignore_index)
    if name == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index if ignore_index >= 0 else -100)
    raise ValueError(f"Loss no soportada: {name}")


def encoder_choices_for_arch(arch, encoders):
    if arch == "UnetPlusPlus":
        return [encoder for encoder in encoders if not encoder.startswith("mit_")]
    return encoders


def make_model(arch, encoder, encoder_weights, num_classes):
    model_cls = ARCH_REGISTRY[arch]
    return model_cls(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )


def parameter_groups(model, decoder_lr, encoder_lr_ratio):
    decoder_params = []
    for attr in ("decoder", "segmentation_head", "classification_head"):
        module = getattr(model, attr, None)
        if module is not None:
            decoder_params.extend(list(module.parameters()))
    decoder_param_ids = {id(param) for param in decoder_params}
    encoder_params = [param for param in model.parameters() if id(param) not in decoder_param_ids]
    return [
        {"params": encoder_params, "lr": decoder_lr * encoder_lr_ratio},
        {"params": decoder_params, "lr": decoder_lr},
    ]


def set_encoder_trainable(model, trainable):
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return
    for param in encoder.parameters():
        param.requires_grad = trainable


def update_confusion(stats, logits, target, num_classes, ignore_index, include_background):
    pred = torch.argmax(logits, dim=1)
    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_index >= 0:
        valid = target != ignore_index
    start = 0 if include_background else 1
    with torch.no_grad():
        for class_id in range(start, num_classes):
            pred_c = (pred == class_id) & valid
            target_c = (target == class_id) & valid
            inter = torch.logical_and(pred_c, target_c).sum().item()
            union = torch.logical_or(pred_c, target_c).sum().item()
            pred_sum = pred_c.sum().item()
            target_sum = target_c.sum().item()
            stats[class_id]["intersection"] += inter
            stats[class_id]["union"] += union
            stats[class_id]["pred"] += pred_sum
            stats[class_id]["target"] += target_sum


def summarize_metrics(stats, include_background):
    ious = []
    dices = []
    per_class = {}
    for class_id, item in stats.items():
        inter = item["intersection"]
        union = item["union"]
        denom = item["pred"] + item["target"]
        if union > 0:
            iou = inter / union
            ious.append(iou)
        else:
            iou = None
        if denom > 0:
            dice = 2.0 * inter / denom
            dices.append(dice)
        else:
            dice = None
        per_class[str(class_id)] = {"iou": iou, "dice": dice}
    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_dice = float(np.mean(dices)) if dices else 0.0
    return {
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "per_class": per_class,
        "include_background": include_background,
    }


def run_epoch(model, loader, criterion, optimizer, device, train, num_classes, ignore_index, include_background, max_batches=0):
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    stats = {
        class_id: {"intersection": 0, "union": 0, "pred": 0, "target": 0}
        for class_id in range(0 if include_background else 1, num_classes)
    }
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch_idx, (images, masks, _) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, masks)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1
            if not train:
                update_confusion(stats, logits, masks, num_classes, ignore_index, include_background)
            if max_batches and batch_idx >= max_batches:
                break

    metrics = summarize_metrics(stats, include_background)
    return total_loss / max(1, n_batches), metrics


def estimate_latency_ms(model, loader, device, repeats=8):
    try:
        images, _, _ = next(iter(loader))
    except StopIteration:
        return 0.0
    sample = images[:1].to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(2):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / max(1, repeats)


def write_csv(path, rows):
    fields = ["epoch", "phase", "train_loss", "val_loss", "val_mean_iou", "val_mean_dice", "score"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def colorize_mask(mask, palette):
    clipped = np.clip(mask, 0, len(palette) - 1).astype(np.int64)
    return palette[clipped][:, :, ::-1]


def save_visuals(model, loader, device, output_dir, palette, max_samples=4):
    vis_dir = Path(output_dir) / "vis_samples"
    vis_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0
    with torch.no_grad():
        for images, masks, paths in loader:
            logits = model(images.to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            gts = masks.numpy()
            for idx, image_path in enumerate(paths):
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is None:
                    continue
                h, w = preds[idx].shape
                image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

                gt_color = colorize_mask(np.where(gts[idx] < 0, 0, gts[idx]), palette)
                pred_color = colorize_mask(preds[idx], palette)
                gt_overlay = cv2.addWeighted(image, 0.55, gt_color, 0.45, 0)
                pred_overlay = cv2.addWeighted(image, 0.55, pred_color, 0.45, 0)
                canvas = np.hstack([image, gt_overlay, pred_overlay])
                cv2.imwrite(str(vis_dir / Path(image_path).name), canvas)
                saved += 1
                if saved >= max_samples:
                    return


def load_search_space(args):
    ss = read_yaml(args.search_space)
    if args.architectures:
        ss["architectures"] = split_csv(args.architectures)
    if args.encoders:
        ss["encoders"] = split_csv(args.encoders)
    if args.losses:
        ss["losses"] = split_csv(args.losses)
    return ss


def suggest_float(trial, ss, name, default_min, default_max, log=False):
    cfg = ss.get(name, {})
    return trial.suggest_float(
        name,
        float(cfg.get("min", default_min)),
        float(cfg.get("max", default_max)),
        log=bool(cfg.get("log", log)),
    )


def suggest_categorical(trial, ss, name, default_values):
    values = ss.get(name, default_values)
    if not isinstance(values, list):
        raise ValueError(f"{name} debe ser lista en search_space.yaml")
    return trial.suggest_categorical(name, values)


def objective(trial, args, ss, train_dataset, val_dataset, device, palette, class_names):
    arch = suggest_categorical(trial, ss, "architectures", ["Unet", "FPN", "DeepLabV3Plus"])
    encoders = encoder_choices_for_arch(arch, ss.get("encoders", ["resnet34", "mobilenet_v2"]))
    if not encoders:
        raise optuna.exceptions.TrialPruned(f"No hay encoders compatibles con {arch}")
    encoder = trial.suggest_categorical(f"encoder_{arch}", encoders)
    loss_name = suggest_categorical(trial, ss, "losses", ["dice_ce", "focal_dice", "tversky_ce"])
    aug_level = suggest_categorical(trial, ss, "augmentations", ["light", "medium"])
    batch_size = suggest_categorical(trial, ss, "batch_sizes", [4, 8])
    decoder_lr = suggest_float(trial, ss, "decoder_lr", 5e-5, 2e-3, log=True)
    encoder_lr_ratio = suggest_float(trial, ss, "encoder_lr_ratio", 0.1, 1.0, log=False)
    weight_decay = suggest_float(trial, ss, "weight_decay", 1e-5, 2e-2, log=True)
    dice_weight = suggest_float(trial, ss, "dice_weight", 0.35, 0.75, log=False)

    trial_dir = Path(args.output_dir) / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    train_dataset.transform = build_train_transform(args.img_size, aug_level)
    val_dataset.transform = build_val_transform(args.img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    encoder_weights = parse_encoder_weights(args.encoder_weights)
    try:
        model = make_model(arch, encoder, encoder_weights, args.num_classes)
    except Exception as exc:
        raise optuna.exceptions.TrialPruned(f"Modelo invalido: {exc}") from exc

    model = model.to(device)
    criterion = make_loss(loss_name, args.num_classes, dice_weight, args.ignore_index, args.include_background_loss)
    optimizer = optim.AdamW(parameter_groups(model, decoder_lr, encoder_lr_ratio), weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-7)

    config = {
        "architecture": arch,
        "encoder": encoder,
        "encoder_weights": encoder_weights,
        "from_scratch": encoder_weights is None,
        "in_channels": 3,
        "classes": args.num_classes,
        "class_names": class_names,
        "activation": None,
        "img_size": args.img_size,
        "loss": loss_name,
        "augmentation": aug_level,
        "ignore_index": args.ignore_index,
        "objective_metric": args.objective_metric,
    }
    (trial_dir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"NAS trial {trial.number}: {arch} + {encoder} | loss={loss_name} | batch={batch_size}")
    print(f"From scratch: {encoder_weights is None} | aug={aug_level} | objective={args.objective_metric}")
    print("=" * 72)

    best_score = -1.0
    best_metrics = {"mean_iou": 0.0, "mean_dice": 0.0}
    best_epoch = 0
    no_improve = 0
    rows = []

    if args.freeze_encoder_phase1:
        set_encoder_trainable(model, False)

    for epoch in range(args.epochs):
        phase = 1
        if args.freeze_encoder_phase1 and epoch == args.phase1_epochs:
            set_encoder_trainable(model, True)
            optimizer = optim.AdamW(parameter_groups(model, decoder_lr, encoder_lr_ratio), weight_decay=weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs - args.phase1_epochs),
                eta_min=1e-7,
            )
            phase = 2
        elif args.freeze_encoder_phase1 and epoch > args.phase1_epochs:
            phase = 2

        train_loss, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            True,
            args.num_classes,
            args.ignore_index,
            args.include_background_metrics,
            args.max_train_batches,
        )
        scheduler.step()
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            False,
            args.num_classes,
            args.ignore_index,
            args.include_background_metrics,
            args.max_val_batches,
        )

        score = val_metrics[args.objective_metric]
        rows.append({
            "epoch": epoch + 1,
            "phase": phase,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_mean_iou": val_metrics["mean_iou"],
            "val_mean_dice": val_metrics["mean_dice"],
            "score": score,
        })
        print(
            f"  [P{phase} E{epoch + 1:03d}] "
            f"TrLoss={train_loss:.4f} ValLoss={val_loss:.4f} "
            f"mIoU={val_metrics['mean_iou']:.4f} mDice={val_metrics['mean_dice']:.4f} "
            f"Score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_metrics = val_metrics
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), trial_dir / "best_model.pth")
        else:
            no_improve += 1

        trial.report(score, epoch)
        if trial.should_prune():
            write_csv(trial_dir / "training_log.csv", rows)
            raise optuna.exceptions.TrialPruned()
        if no_improve >= args.patience:
            print(f"  Early stopping en epoca {epoch + 1}")
            break

    write_csv(trial_dir / "training_log.csv", rows)
    if (trial_dir / "best_model.pth").exists():
        state = torch.load(trial_dir / "best_model.pth", map_location=device, weights_only=True)
        model.load_state_dict(state)
        save_visuals(model, val_loader, device, trial_dir, palette, args.max_visuals)

    latency_ms = estimate_latency_ms(model, val_loader, device)
    params_m = sum(p.numel() for p in model.parameters()) / 1_000_000.0
    final_score = best_score - args.latency_penalty * latency_ms - args.param_penalty * params_m

    trial_info = {
        "trial_number": trial.number,
        "score": final_score,
        "raw_score": best_score,
        "best_mean_iou": best_metrics["mean_iou"],
        "best_mean_dice": best_metrics["mean_dice"],
        "best_epoch": best_epoch,
        "latency_ms_per_tile": latency_ms,
        "params_m": params_m,
        "params": trial.params,
        "model_config": config,
        "per_class": best_metrics.get("per_class", {}),
    }
    (trial_dir / "trial_info.json").write_text(json.dumps(trial_info, indent=2), encoding="utf-8")
    return final_score


def parse_args():
    parser = argparse.ArgumentParser(description="NAS desde cero para segmentacion semantica.")
    parser.add_argument("--data", default="dataset/tiles", help="Dataset con train/valid/test e images/masks.")
    parser.add_argument("--output-dir", default="runs/semantic_nas", help="Directorio de salida.")
    parser.add_argument("--num-classes", type=int, default=2, help="Total de clases incluyendo fondo.")
    parser.add_argument("--class-config", default=None, help="JSON opcional con nombres y colores de clases.")
    parser.add_argument("--search-space", default=str(DEFAULT_SEARCH_SPACE), help="YAML con espacio de busqueda NAS.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--phase1-epochs", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore-index", type=int, default=-1, help="Valor void. Usa 255 solo si NO es foreground.")
    parser.add_argument("--objective-metric", choices=["mean_iou", "mean_dice"], default="mean_iou")
    parser.add_argument("--encoder-weights", default=None, help="Default None: desde cero. Usa 'imagenet' si quieres pretraining.")
    parser.add_argument("--freeze-encoder-phase1", action="store_true", help="Opcional; util con encoder_weights=imagenet.")
    parser.add_argument("--include-background-loss", action="store_true")
    parser.add_argument("--include-background-metrics", action="store_true")
    parser.add_argument("--architectures", default=None, help="Override CSV: Unet,FPN,DeepLabV3Plus")
    parser.add_argument("--encoders", default=None, help="Override CSV: resnet34,mit_b0,...")
    parser.add_argument("--losses", default=None, help="Override CSV: dice_ce,focal_dice,tversky_ce,ce")
    parser.add_argument("--latency-penalty", type=float, default=0.0)
    parser.add_argument("--param-penalty", type=float, default=0.0)
    parser.add_argument("--max-train-batches", type=int, default=0, help="Smoke test.")
    parser.add_argument("--max-val-batches", type=int, default=0, help="Smoke test.")
    parser.add_argument("--max-visuals", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_classes < 2:
        raise ValueError("--num-classes debe incluir fondo y al menos una clase de objeto.")
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ss = load_search_space(args)

    class_names, color_to_id = load_class_config(args.class_config, args.num_classes)
    palette = default_palette(args.num_classes)

    data_dir = Path(args.data)
    train_dataset = SemanticSegmentationDataset(
        data_dir / "train" / "images",
        data_dir / "train" / "masks",
        build_train_transform(args.img_size, "medium"),
        args.num_classes,
        args.ignore_index,
        color_to_id,
    )
    val_dataset = SemanticSegmentationDataset(
        data_dir / "valid" / "images",
        data_dir / "valid" / "masks",
        build_val_transform(args.img_size),
        args.num_classes,
        args.ignore_index,
        color_to_id,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(f"Dataset vacio o estructura incorrecta: {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {data_dir} | train={len(train_dataset)} valid={len(val_dataset)}")
    print(f"Clases ({args.num_classes}): {class_names}")
    print(f"Encoder weights: {parse_encoder_weights(args.encoder_weights)}")
    print(f"Salida: {output_dir}")

    study = optuna.create_study(
        study_name="semantic_segmentation_nas_from_scratch",
        storage=f"sqlite:///{output_dir / 'nas_study.db'}",
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=5),
    )
    study.optimize(
        lambda trial: objective(trial, args, ss, train_dataset, val_dataset, device, palette, class_names),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    best_trial_dir = output_dir / f"trial_{study.best_trial.number}"
    summary = {
        "best_score": study.best_trial.value,
        "best_trial": study.best_trial.number,
        "best_checkpoint": str(best_trial_dir / "best_model.pth"),
        "best_model_config": str(best_trial_dir / "model_config.json"),
        "params": study.best_trial.params,
    }
    (output_dir / "best_nas_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nNAS completado")
    print(f"  Mejor score: {study.best_trial.value:.4f}")
    print(f"  Mejor trial: {study.best_trial.number}")
    print(f"  Checkpoint: {best_trial_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
