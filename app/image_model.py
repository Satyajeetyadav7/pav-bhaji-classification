"""Pixel-side classifier.

The notebook's model is deliberately text-only, so it cannot say anything about
an image it has never seen. For the upload UI we need something that reads
actual pixels, trained on the 452 hand-labelled images in dataset/images/.

Only PIL + numpy are available here (no torchvision), so this is classical
computer vision rather than a CNN: colour statistics, a spatial grid, and edge
/ texture descriptors. Pav bhaji is visually distinctive - a saturated
orange-red mash, usually centred, with pale bread and a butter cube - so colour
geometry carries a real signal.
"""
import numpy as np
from PIL import Image, ImageFilter

IMG_SIZE = 144
GRID = 3

# PIL hue is 0-255, not 0-360.
HUE_BHAJI = (0, 24)        # red -> orange
HUE_BUTTER = (24, 45)      # yellow
HUE_GREEN = (55, 110)      # coriander / salad / green chutney


def _load(img):
    """Accept a path, a file-like object or a PIL image; return RGB, square."""
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    img = img.convert("RGB")
    return img


def _block_stats(arr, grid=GRID):
    """Mean and std of each channel over a grid x grid tiling."""
    h, w, c = arr.shape
    hs, ws = h // grid, w // grid
    out = []
    for i in range(grid):
        for j in range(grid):
            tile = arr[i * hs:(i + 1) * hs, j * ws:(j + 1) * ws]
            out.extend(tile.reshape(-1, c).mean(axis=0))
            out.extend(tile.reshape(-1, c).std(axis=0))
    return out


def _hue_mass(hsv, lo, hi, s_min=70, v_min=40):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (h >= lo) & (h < hi) & (s >= s_min) & (v >= v_min)
    return float(m.mean())


def extract_features(img):
    img0 = _load(img)
    aspect = img0.width / max(img0.height, 1)

    im = img0.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    rgb = np.asarray(im, dtype=np.float32)
    hsv = np.asarray(im.convert("HSV"), dtype=np.float32)
    gray = np.asarray(im.convert("L"), dtype=np.float32)

    f = []

    # --- global colour histograms ---------------------------------------
    for chan, bins, rng in ((hsv[..., 0], 36, (0, 256)),
                            (hsv[..., 1], 12, (0, 256)),
                            (hsv[..., 2], 12, (0, 256))):
        hist, _ = np.histogram(chan, bins=bins, range=rng)
        f.extend(hist / chan.size)

    # --- global moments --------------------------------------------------
    for arr in (rgb, hsv):
        flat = arr.reshape(-1, 3)
        f.extend(flat.mean(axis=0) / 255.0)
        f.extend(flat.std(axis=0) / 255.0)

    # --- spatial grid ----------------------------------------------------
    f.extend(np.asarray(_block_stats(hsv)) / 255.0)
    f.extend(np.asarray(_block_stats(rgb)) / 255.0)

    # --- centre vs border (the dish usually sits in the middle) ----------
    c0, c1 = int(IMG_SIZE * 0.25), int(IMG_SIZE * 0.75)
    centre = hsv[c0:c1, c0:c1]
    mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=bool)
    mask[c0:c1, c0:c1] = False
    border = hsv[mask]
    f.extend((centre.reshape(-1, 3).mean(axis=0) - border.mean(axis=0)) / 255.0)

    # --- dish-specific colour mass --------------------------------------
    for lo, hi in (HUE_BHAJI, HUE_BUTTER, HUE_GREEN):
        f.append(_hue_mass(hsv, lo, hi))
        f.append(_hue_mass(centre, lo, hi))

    # warm, saturated, bright: the bhaji signature
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    f.append(float(((h < 30) & (s > 120)).mean()))
    f.append(float((s > 150).mean()))
    f.append(float((v > 200).mean()))          # blown highlights / white plate
    f.append(float((v < 50).mean()))           # dark background
    f.append(float(s.mean() / 255.0))
    f.append(float(v.mean() / 255.0))

    # --- texture / edges --------------------------------------------------
    edges = np.asarray(im.convert("L").filter(ImageFilter.FIND_EDGES),
                       dtype=np.float32)
    f.append(float(edges.mean() / 255.0))
    f.append(float(edges.std() / 255.0))
    f.append(float((edges > 40).mean()))
    gy, gx = np.gradient(gray)
    f.append(float(np.hypot(gx, gy).mean() / 255.0))

    # colour diversity: a plated single dish uses fewer distinct colours than a
    # collage or a menu board
    q = (rgb // 32).astype(np.int32)
    codes = q[..., 0] * 64 + q[..., 1] * 8 + q[..., 2]
    f.append(len(np.unique(codes)) / 512.0)

    f.append(aspect)
    f.append(float(abs(aspect - 1.0) < 0.02))

    return np.asarray(f, dtype=np.float32)


def feature_dim():
    return len(extract_features(Image.new("RGB", (64, 64), (128, 90, 40))))


def image_model():
    """Classifier over the descriptor above."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  ExtraTreesClassifier, VotingClassifier)

    lr = Pipeline([("sc", StandardScaler()),
                   ("clf", LogisticRegression(C=0.1, max_iter=5000,
                                              class_weight="balanced",
                                              random_state=42))])
    et = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=2,
                              class_weight="balanced", n_jobs=-1,
                              random_state=42)
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                         max_leaf_nodes=15,
                                         l2_regularization=1.0,
                                         random_state=42)
    return VotingClassifier([("lr", lr), ("et", et), ("hgb", hgb)], voting="soft")
