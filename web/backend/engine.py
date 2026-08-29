# -*- coding: utf-8 -*-
"""
engine.py — pure-Python clinical engine for the web version.

Ports the core logic of main46.py (PyQt5 app) to framework-free functions:
  * volume rotation (Z/X axis, chained)
  * auto missing-tooth detection (EfficientNet-B0 2.5D classifier)
  * YOLOv8 segmentation of alveolar bone (AB) and mandibular canal (MC)
  * implant recommendation algorithm (middle-slice precheck + full-slice check)
  * applying a chosen implant (cylindrical mask + overlay color)

Margins stay identical to the desktop app:
  1 mm safety zone, 2 mm top margin to canal, 3 mm band below crest,
  0.3 mm/pixel.
"""
from __future__ import annotations

import math
import os
import re
import shutil
from collections import OrderedDict

import cv2
import numpy as np
import scipy.ndimage as ndi

PIXEL_SIZE_MM = 0.3
_CLF = dict(img_size=224, k=2, missing_label=0, use_clahe=False)

# ---------------------------------------------------------------- helpers
def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def slice_to_uint8(slc):
    s = np.asarray(slc)
    if s.dtype != np.uint8:
        vmin, vmax = float(s.min()), float(s.max())
        if vmax > vmin:
            s = ((s - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
        else:
            s = np.zeros_like(s, dtype=np.uint8)
    return s

def convert_to_pixels(mm):
    return int(mm / PIXEL_SIZE_MM)

# ------------------------------------------------------------- rotation
def rotate_volume(volume, angle, axis="z"):
    axes = (1, 2) if axis.lower().startswith("z") else (0, 1)
    return ndi.rotate(volume, angle=float(angle), axes=axes, reshape=True)

# ---------------------------------------------------------- classifier
def prep_single_slice(slc2d):
    img = np.asarray(slc2d)
    if img.dtype != np.uint8:
        vmin, vmax = float(np.min(img)), float(np.max(img))
        if vmax > vmin:
            img = ((img - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    roi = img[:, :]
    if _CLF["use_clahe"]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        roi = clahe.apply(roi)
    roi = cv2.resize(roi, (_CLF["img_size"], _CLF["img_size"]), interpolation=cv2.INTER_AREA)
    roi = roi.astype(np.float32)
    mn, mx = float(roi.min()), float(roi.max())
    roi = (roi - mn) / (mx - mn) if (mx - mn) > 1e-6 else roi
    return roi

def slice_stack_2p5d(vol, idx, k=2):
    Z = vol.shape[0]
    chans = []
    for d in range(-k, k + 1):
        j = min(max(idx + d, 0), Z - 1)
        chans.append(prep_single_slice(vol[j]))
    return np.stack(chans, axis=0)

def load_classifier(model_path, device="cpu"):
    import torch
    torch.set_num_threads(1)
    import torch.nn as nn
    from torchvision import models

    try:
        m = torch.jit.load(model_path, map_location=device)
        m.eval()
        return m
    except Exception:
        pass

    obj = torch.load(model_path, map_location=device)
    if hasattr(obj, "forward"):
        try:
            obj.eval()
        except Exception:
            pass
        return obj

    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    else:
        sd = obj if isinstance(obj, (dict, OrderedDict)) else None
    if sd is None:
        raise RuntimeError("Unsupported model format in classification.pt")

    def strip_prefix(state_dict, prefixes=("net.", "module.", "model.")):
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            nk = k
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
            new_sd[new_sd.setdefault(nk, nk)] = v
        return new_sd

    sd = strip_prefix(sd)
    in_ch = int(2 * _CLF["k"] + 1)
    base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    old_first = base.features[0][0]
    new_first = nn.Conv2d(in_ch, old_first.out_channels,
                          kernel_size=old_first.kernel_size, stride=old_first.stride,
                          padding=old_first.padding, bias=False)
    with torch.no_grad():
        w = old_first.weight
        w_mean = w.mean(dim=1, keepdim=True)
        new_first.weight.copy_(w_mean.repeat(1, in_ch, 1, 1) if in_ch != 3 else w)
    base.features[0][0] = new_first
    in_feat = base.classifier[1].in_features
    base.classifier[1] = nn.Linear(in_feat, 2)
    try:
        base.load_state_dict(sd, strict=True)
    except Exception:
        model_keys = set(base.state_dict().keys())
        filtered = OrderedDict((k, v) for k, v in sd.items() if k in model_keys)
        base.load_state_dict(filtered, strict=False)
    base.eval()
    return base

def classify_volume_missing_range(vol, model, device="cpu", progress_cb=None):
    import torch
    model.eval()
    Z = vol.shape[0]
    preds = []
    with torch.inference_mode():
        for i in range(Z):
            x_np = slice_stack_2p5d(vol, i, k=_CLF["k"])
            x = torch.from_numpy(x_np).unsqueeze(0).to(device, dtype=torch.float32)
            logits = model(x)
            preds.append(int(torch.argmax(logits, dim=1).item()))
            if progress_cb and (i % 10 == 0 or i == Z - 1):
                progress_cb((i + 1) / Z)
    miss_lbl = int(_CLF["missing_label"])
    idx = [i for i, p in enumerate(preds) if p == miss_lbl]
    if not idx:
        alt = [i for i, p in enumerate(preds) if p == 1 - miss_lbl]
        if alt:
            return min(alt), max(alt), preds
        return None, None, preds
    return min(idx), max(idx), preds

# ------------------------------------------------------ YOLO segmentation
def run_yolo_segmentation(vol, s_val, e_val, ab_dir, mc_dir, model_path, progress_cb=None):
    for d in (ab_dir, mc_dir):
        os.makedirs(d, exist_ok=True)
        try:
            for name in os.listdir(d):
                fp = os.path.join(d, name)
                if os.path.isfile(fp) or os.path.islink(fp):
                    os.remove(fp)
                else:
                    shutil.rmtree(fp, ignore_errors=True)
        except Exception:
            pass

    from ultralytics import YOLO
    import torch

    H, W = vol.shape[1], vol.shape[2]
    src_indices = list(range(s_val, e_val + 1))
    imgs_640 = []
    for idx in src_indices:
        rgb = cv2.merge((slice_to_uint8(vol[idx]),) * 3)
        imgs_640.append(cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_AREA))

    model = YOLO(model_path, task="segment")
    results = model(imgs_640, save=False, imgsz=640, verbose=False)

    i = s_val
    for res in results:
        ab_bin = np.zeros((H, W), dtype=np.uint8)
        mc_bin = None
        if getattr(res, "masks", None) is not None and res.masks is not None:
            try:
                boxes = res.boxes.data if res.boxes is not None else None
                clss = boxes[:, 5] if (boxes is not None and boxes.numel() > 0) else torch.empty((0,))
                ab_idx = torch.where(clss == 0)[0]
                mc_idx = torch.where(clss == 1)[0]
                mdata = res.masks.data
                if mdata is not None and mdata.numel() > 0:
                    if ab_idx.numel() > 0:
                        ab_stack = mdata[ab_idx].cpu().numpy()
                        ab_union = (ab_stack > 0.5).any(axis=0).astype(np.uint8) * 255
                        _, ab_thresh = cv2.threshold(ab_union, 127, 255, cv2.THRESH_BINARY)
                        cnts, _ = cv2.findContours(ab_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        ab_filled_640 = np.zeros_like(ab_union)
                        cv2.drawContours(ab_filled_640, cnts, -1, 255, thickness=cv2.FILLED)
                        ab_bin = cv2.resize(ab_filled_640, (W, H), interpolation=cv2.INTER_NEAREST)
                    if mc_idx.numel() > 0:
                        mc_stack = mdata[mc_idx].cpu().numpy()
                        mc_union = (mc_stack > 0.5).any(axis=0).astype(np.uint8) * 255
                        mc_bin = cv2.resize(mc_union, (W, H), interpolation=cv2.INTER_NEAREST)
            except Exception:
                pass

        ab_bin = np.asarray(ab_bin) if ab_bin is not None else np.zeros((H, W), dtype=np.uint8)
        mc_bin = np.asarray(mc_bin) if mc_bin is not None else np.zeros((H, W), dtype=np.uint8)
        if ab_bin.ndim != 2:
            ab_bin = np.zeros((H, W), dtype=np.uint8)
        if mc_bin.ndim != 2:
            mc_bin = np.zeros((H, W), dtype=np.uint8)
        if ab_bin.shape != (H, W):
            ab_bin = cv2.resize(ab_bin, (W, H), interpolation=cv2.INTER_NEAREST)
        if mc_bin.shape != (H, W):
            mc_bin = cv2.resize(mc_bin, (W, H), interpolation=cv2.INTER_NEAREST)
        ab_bin = ab_bin.astype(np.uint8)
        mc_bin = mc_bin.astype(np.uint8)

        ab_union_mc = cv2.bitwise_or(ab_bin, mc_bin)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        ab_closed = cv2.morphologyEx(ab_union_mc, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        ab_union_mc = (ndi.binary_fill_holes(ab_closed > 0).astype(np.uint8)) * 255

        cv2.imwrite(os.path.join(ab_dir, f"merged_segs_{i}.jpg"), ab_union_mc)
        cv2.imwrite(os.path.join(mc_dir, f"merged_segs_{i}.jpg"), mc_bin)
        i += 1
        if progress_cb:
            progress_cb((i - s_val) / max(1, len(src_indices)))
    return i - s_val

# ------------------------------------------------- recommendation algorithm
def _binarize(gray):
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    filled = ndi.binary_fill_holes(binary > 0)
    return filled.astype(np.uint8) * 255

def _largest_contour(cnts):
    if not cnts:
        return None
    areas = [cv2.contourArea(c) for c in cnts]
    return cnts[int(np.argmax(areas))]

def _load_mask_stack(path):
    imgs = []
    if not os.path.isdir(path):
        return None
    for img in sorted(os.listdir(path), key=natural_key):
        if img.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            im = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                imgs.append(_binarize(im))
    return imgs or None

def _segmented_area(image):
    binary_image = _binarize(image)
    contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = _largest_contour(contours)
        x, y, w, h = cv2.boundingRect(c)
        return x, y, w, h, [c]
    return None, None, None, None, None

def _canal_position(image):
    binary_image = _binarize(image)
    contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = _largest_contour(contours)
        x, y, w, h = cv2.boundingRect(c)
        return y, y + h
    return None

def _volume_cyl(d_mm, h_mm):
    r = d_mm / 2.0
    return 3.14159 * (r ** 2) * h_mm

def _find_best_implant_middle(alv_imgs, mc_imgs, imps):
    def calculate_volume(d, h):
        r = d / 2
        return 3.14159 * (r ** 2) * h

    reasons = {}
    number = len(alv_imgs)
    if number == 0:
        for rid, d, h in imps:
            reasons[rid] = ("no AB/MC images available in the middle-slice precheck "
                            "(empty segmentation stack for this gap).")
        return None, None, reasons

    middle_index = number // 2 if number % 2 == 0 else number // 2 + 1
    if middle_index >= number:
        middle_index = number // 2

    safety_zone_pixels = convert_to_pixels(1)
    top_margin_pixels = convert_to_pixels(2)
    line_margin_3mm_pixels = convert_to_pixels(3)
    thickness_to_diameter = len(alv_imgs) * PIXEL_SIZE_MM
    fitting = []

    alveolar_bone = _binarize(alv_imgs[middle_index])
    mandibular_canal = _binarize(mc_imgs[middle_index])
    x_ab, y_ab, w_ab, h_ab, contours_ab = _segmented_area(alveolar_bone)
    canal_position = _canal_position(mandibular_canal)

    if (x_ab is None) or (canal_position is None):
        for rid, d, h in imps:
            reasons[rid] = ("no valid AB/MC segmentation on the middle slice "
                            "(cannot locate alveolar bone or mandibular canal).")
        return None, None, reasons

    canal_top, _ = canal_position
    below_3mm_points = [pt[0] for pt in contours_ab[0] if pt[0][1] >= y_ab + line_margin_3mm_pixels]
    if not below_3mm_points:
        for rid, d, h in imps:
            reasons[rid] = ("no usable 3 mm band below the crest in the middle slice "
                            "(insufficient crest region for safe placement).")
        return None, None, reasons

    top_contour_points = [pt[0][0] for pt in contours_ab[0] if pt[0][1] == y_ab]
    center_top_x = (sum(top_contour_points) // len(top_contour_points)
                    if top_contour_points else (x_ab + w_ab // 2))

    iter_erosion = max(1, convert_to_pixels(1))
    eroded_ab = cv2.erode(alveolar_bone, np.ones((safety_zone_pixels, safety_zone_pixels), np.uint8),
                          iterations=iter_erosion)
    H, W = alveolar_bone.shape[:2]

    for rid, diameter, height in imps:
        d_px = convert_to_pixels(diameter)
        h_px = convert_to_pixels(height)
        if diameter > thickness_to_diameter:
            reasons[rid] = (f"diameter {diameter:g} mm exceeds the available alveolar bone "
                            f"thickness (~{thickness_to_diameter:.2f} mm) in the middle slice.")
            continue
        implant_x = max(0, min(center_top_x - (d_px // 2), W - d_px))
        implant_y = y_ab
        ys = y_ab + line_margin_3mm_pixels
        ye = min(ys + h_px, H)
        if implant_x < 0 or implant_x + d_px > W or ys < 0 or ye > H:
            reasons[rid] = (f"implant footprint (D={diameter:g} mm, H={height:g} mm) "
                            f"goes outside the eroded AB safety zone in the middle slice.")
            continue
        area = eroded_ab[ys:ye, implant_x:implant_x + d_px]
        if area.size == 0 or not np.all(area == 255):
            reasons[rid] = (f"diameter {diameter:g} mm is too wide to fit inside the eroded "
                            f"AB safety zone in the middle slice while keeping a 3 mm band "
                            f"below the crest for H={height:g} mm.")
            continue
        found_pos = False
        for dy in range(implant_y, canal_top - top_margin_pixels - h_px + 1):
            area_full = alveolar_bone[dy:dy + h_px, implant_x:implant_x + d_px]
            if area_full.size > 0 and np.all(area_full == 255):
                implant_y = dy
                found_pos = True
                break
        if not found_pos:
            reasons[rid] = (f"no vertical position in the middle slice where "
                            f"H={height:g} mm can be fully contained in AB while "
                            "remaining above the mandibular canal.")
            continue
        if implant_y + h_px >= canal_top - top_margin_pixels:
            reasons[rid] = (f"too close to the mandibular canal in the middle slice; "
                            f"H={height:g} mm violates the 2 mm safety margin.")
            continue
        vol = calculate_volume(diameter, height)
        fitting.append((rid, diameter, height, vol, implant_x, implant_y))

    if not fitting:
        return None, middle_index, reasons
    fitting.sort(key=lambda x: x[3], reverse=True)
    return fitting, middle_index, reasons

def _find_best_implant(alv_imgs, mc_imgs, imps, start_slice, end_slice):
    logs = []
    mid_imps, mid_idx, mid_reasons = _find_best_implant_middle(alv_imgs, mc_imgs, imps)
    if mid_imps is None:
        logs.append("Middle-slice precheck failed for all implants. See per-implant reasons below.")
        for rid, d, h in imps:
            reason_txt = mid_reasons.get(rid, "rejected at middle-slice precheck "
                                              "(unspecified AB/MC geometry constraint).")
            logs.append(f"{rid} (D={d} mm, H={h} mm): {reason_txt}")
        return [], logs

    mid_ids = {rid for (rid, _, _, _, _, _) in mid_imps}
    for rid, d, h in imps:
        if rid not in mid_ids:
            reason_txt = mid_reasons.get(rid, "rejected at middle-slice precheck "
                                              "(implant cannot be safely positioned in the middle slice "
                                              "given AB thickness and canal position).")
            logs.append(f"{rid} (D={d} mm, H={h} mm): {reason_txt}")

    valid = []
    Z = len(alv_imgs)
    length_bt = end_slice - start_slice + 1
    safety_zone_pixels = convert_to_pixels(1)
    top_margin_pixels = convert_to_pixels(2)
    line_margin_3mm_pixels = convert_to_pixels(3)
    safety_zone_slices = math.ceil(1 * PIXEL_SIZE_MM / PIXEL_SIZE_MM)

    for rid, diameter, height, volume_mid, initial_x, initial_y in mid_imps:
        reason = None
        d_sl = math.ceil(diameter / PIXEL_SIZE_MM)
        if d_sl + safety_zone_slices > length_bt:
            reason = (f"{rid} (D={diameter} mm, H={height} mm): "
                      f"diameter {diameter:g} mm too large for the selected gap "
                      f"(thickness ~{length_bt * PIXEL_SIZE_MM:.2f} mm; needs at least "
                      f"{(d_sl + safety_zone_slices) * PIXEL_SIZE_MM:.2f} mm including safety zone).")
            logs.append(reason)
            continue
        start_i = max(0, mid_idx - (d_sl // 2) - (0 if d_sl % 2 else 1))
        end_i = min(Z - 1, mid_idx + (d_sl // 2))
        if start_i > end_i:
            logs.append(f"{rid}: computed slice range invalid after clamping (start_i={start_i}, end_i={end_i}).")
            continue
        d_px = convert_to_pixels(diameter)
        h_px = convert_to_pixels(height)
        suitable = True
        for i in range(start_i, end_i + 1):
            alv = _binarize(alv_imgs[i])
            mc = _binarize(mc_imgs[i])
            x_ab, y_ab, w_ab, h_ab, contours_ab = _segmented_area(alv)
            canal_position = _canal_position(mc)
            if None in (x_ab, y_ab, w_ab, h_ab) or (canal_position is None):
                suitable = False
                reason = (f"{rid} (D={diameter} mm, H={height} mm): "
                          f"missing or invalid AB/MC segmentation at slice "
                          f"(relative {i}, global {i + start_slice}).")
                break
            canal_top, _ = canal_position
            eroded = cv2.erode(alv, np.ones((safety_zone_pixels, safety_zone_pixels), np.uint8),
                               iterations=max(1, convert_to_pixels(1)))
            ys = y_ab + line_margin_3mm_pixels
            ye = min(ys + h_px, eroded.shape[0])
            xe = min(initial_x + d_px, eroded.shape[1])
            if initial_x < 0 or xe > eroded.shape[1] or ys < 0 or ye > eroded.shape[0]:
                suitable = False
                reason = (f"{rid} (D={diameter} mm, H={height} mm): "
                          f"implant rectangle goes outside alveolar bone "
                          f"at slice (relative {i}, global {i + start_slice}).")
                break
            area = eroded[ys:ye, initial_x:xe]
            if area.size == 0 or not np.all(area == 255):
                suitable = False
                reason = (f"{rid} (D={diameter} mm, H={height} mm): "
                          f"eroded AB safety zone not fully available "
                          f"at slice (relative {i}, global {i + start_slice}).")
                break
            if initial_y + h_px >= canal_top - top_margin_pixels:
                suitable = False
                reason = (f"{rid} (D={diameter} mm, H={height} mm): "
                          f"too close to mandibular canal at slice "
                          f"(relative {i}, global {i + start_slice}); "
                          f"violates 2 mm safety margin.")
                break
        if suitable:
            valid.append((rid, diameter, height, initial_x, initial_y, start_i, end_i))
            logs.append(f"{rid} (D={diameter} mm, H={height} mm): OK — "
                        f"fits all slices from global slice "
                        f"{start_i + start_slice} to {end_i + start_slice}.")
        else:
            if reason is None:
                reason = f"{rid} (D={diameter} mm, H={height} mm): rejected for unspecified geometric reason."
            logs.append(reason)
    return valid, logs

def _infer_no_implant_reason(logs):
    height_keywords = (
        "too close to mandibular canal",
        "no usable 3 mm band below the crest",
        "insufficient crest region",
        "no vertical position in the middle slice",
        "can be fully contained in AB",
        "violates the 2 mm safety margin",
    )
    has_height = any(any(kw in line for kw in height_keywords) for line in logs)
    has_diam = any(("diameter" in line) and not any(kw in line for kw in height_keywords) for line in logs)
    if has_diam and not has_height:
        return "sharp"
    if has_height:
        return "short"
    return None

def compute_recommendation(ab_dir, mc_dir, start, end, implants):
    alv_imgs = _load_mask_stack(ab_dir)
    mc_imgs = _load_mask_stack(mc_dir)
    if not alv_imgs or not mc_imgs:
        raise RuntimeError("No segmented AB/MC images found. Make sure segmentation succeeded.")
    if not implants:
        raise RuntimeError("Implant database is empty.")
    best_implants, selection_logs = _find_best_implant(alv_imgs, mc_imgs, implants, start, end)
    no_implant_reason = _infer_no_implant_reason(selection_logs) if not best_implants else None
    cand_with_vol = [(rid, d, h, x, y, si, ei, _volume_cyl(d, h))
                     for (rid, d, h, x, y, si, ei) in best_implants]
    cand_with_vol.sort(key=lambda t: t[-1], reverse=True)
    return {"candidates": cand_with_vol, "logs": selection_logs, "no_implant_reason": no_implant_reason}

# ------------------------------------------------------------- apply implant
def draw_implant_in_images(diameter, height, implant_x, implant_y, out_folder, z0, z1, resolution):
    d_px = convert_to_pixels(diameter)
    h_px = convert_to_pixels(height)
    W, H = resolution
    cx = implant_x + d_px // 2
    z0, z1 = int(z0), int(z1)
    zc = 0.5 * (z0 + z1)
    r = d_px / 2.0
    y0 = max(0, min(implant_y, H - h_px))
    y1 = min(H, y0 + h_px)
    os.makedirs(out_folder, exist_ok=True)
    for z in range(z0, z1 + 1):
        dz = abs(z - zc)
        if dz > r:
            continue
        half_w = int(round((r * r - dz * dz) ** 0.5))
        wz = 2 * half_w
        if wz <= 0:
            continue
        x0 = max(0, min(int(cx - half_w), W))
        x1 = max(0, min(int(cx + half_w), W))
        if x1 <= x0 or y1 <= y0:
            continue
        implant_image = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(implant_image, (x0, y0), (x1, y1), 255, -1)
        cv2.imwrite(os.path.join(out_folder, f"implant_{z}.png"), implant_image)

def _fill_black(total_slices, range_start, range_end, fmt, out_folder, resolution):
    W, H = resolution
    for num in range(total_slices):
        if not (range_start <= num <= range_end):
            image = np.zeros((H, W), dtype=np.uint8)
            cv2.imwrite(os.path.join(out_folder, fmt.format(num=num)), image)

def apply_implant(patient_dir, candidate, start, resolution, total_slices, ab_dir, mc_dir):
    rid, d_mm, h_mm, x_px, y_px, rel_start, rel_end, _vol = candidate
    abs_start = rel_start + start
    abs_end = rel_end + start
    implant_only_path = os.path.join(patient_dir, "implant_only")
    os.makedirs(implant_only_path, exist_ok=True)
    for name in os.listdir(implant_only_path):
        try:
            os.remove(os.path.join(implant_only_path, name))
        except Exception:
            pass
    draw_implant_in_images(d_mm, h_mm, x_px, y_px, implant_only_path,
                           abs_start, abs_end, resolution=resolution)
    _fill_black(total_slices, abs_start, abs_end, fmt="implant_{num}.png",
                out_folder=implant_only_path, resolution=resolution)
    patient_path_imp = os.path.join(patient_dir, "final_implant.npy")
    files = sorted((f for f in os.listdir(implant_only_path) if f.lower().endswith(".png")),
                   key=natural_key)
    arrs = []
    for f in files:
        im = cv2.imread(os.path.join(implant_only_path, f), cv2.IMREAD_GRAYSCALE)
        if im is not None:
            arrs.append(im)
    if arrs:
        np.save(patient_path_imp, np.stack(arrs, axis=0))
    return {"applied": {"id": rid, "diameter_mm": d_mm, "height_mm": h_mm,
                        "start_slice": abs_start, "end_slice": abs_end,
                        "volume_mm3": _vol,
                        # in-plane pixel origin, so the viewer can place a
                        # true-to-scale 3D model at the planned position
                        "x_px": int(x_px), "y_px": int(y_px),
                        "pixel_size_mm": PIXEL_SIZE_MM}}

# ------------------------------------------------- napari-style label volumes
_MASK_SPEC = {
    # kind    -> (folder,          filename template)
    "ab":       ("segmented_AB",  "merged_segs_{z}.jpg"),
    "mc":       ("segmented_MC",  "merged_segs_{z}.jpg"),
    "implant":  ("implant_only",  "implant_{z}.png"),
}

def mask_available(patient_dir, kind):
    spec = _MASK_SPEC.get(kind)
    if spec is None:
        return False
    d = os.path.join(patient_dir, spec[0])
    if not os.path.isdir(d):
        return False
    return any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in os.listdir(d))

def build_label_volume(patient_dir, kind, total_slices):
    """Full (Z, H, W) uint8 0/1 stack for 'ab' | 'mc' | 'implant'.

    Mirrors `_ensure_final_ab_mc_npys` in the desktop app: slices the
    segmentation never produced stay black, so the stack always lines up
    with the base volume along Z.
    """
    spec = _MASK_SPEC.get(kind)
    if spec is None:
        raise ValueError(f"unknown mask kind '{kind}'")
    folder, tmpl = spec
    d = os.path.join(patient_dir, folder)
    if not os.path.isdir(d):
        return None

    # discover H, W from the first readable mask
    H = W = None
    for fn in sorted(os.listdir(d), key=natural_key):
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            im = cv2.imread(os.path.join(d, fn), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                H, W = im.shape[:2]
                break
    if H is None:
        return None

    out = np.zeros((int(total_slices), H, W), dtype=np.uint8)
    found = False
    for z in range(int(total_slices)):
        fp = os.path.join(d, tmpl.format(z=z))
        if not os.path.isfile(fp):
            continue
        im = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        if im.shape[:2] != (H, W):
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_NEAREST)
        out[z] = (im > 127).astype(np.uint8)
        found = True
    return out if found else None
