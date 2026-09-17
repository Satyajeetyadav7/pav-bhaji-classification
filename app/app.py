"""Pav Bhaji classifier - upload an image, get a verdict.

    streamlit run app/app.py

Two models answer every upload:

* the **image model** reads the pixels (colour geometry + texture), trained on
  the 452 hand-labelled photos. It works on any image you throw at it.
* the **caption model** is the notebook's model - it never looks at pixels, it
  reads the Instagram caption, hashtags and metadata. It can only speak when the
  uploaded file is recognised as one of the 1,495 posts in the dataset.
"""
import hashlib
import io
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from joblib import load

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pavbhaji_lib as L
from image_model import extract_features

MODELDIR = os.path.join(L.ROOT, "models")

st.set_page_config(page_title="Pav Bhaji Classifier", page_icon="🍛",
                   layout="centered")

st.markdown("""
<style>
.verdict {border-radius:14px; padding:22px 26px; margin:14px 0 6px 0;
          border:1px solid rgba(128,128,128,.25);}
.verdict h2 {margin:0; font-size:1.7rem; letter-spacing:-.01em;}
.verdict p  {margin:.35rem 0 0 0; opacity:.75; font-size:.92rem;}
.yes {background:rgba(209,73,91,.12); border-color:rgba(209,73,91,.45);}
.no  {background:rgba(48,99,142,.12); border-color:rgba(48,99,142,.45);}
.src {font-size:.78rem; text-transform:uppercase; letter-spacing:.09em;
      opacity:.6; margin-bottom:.2rem;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner=False)
def load_models():
    missing = [f for f in ("image_model.joblib", "text_model.joblib", "posts.pkl")
               if not os.path.exists(os.path.join(MODELDIR, f))]
    if missing:
        return None
    img = load(os.path.join(MODELDIR, "image_model.joblib"))
    txt = load(os.path.join(MODELDIR, "text_model.joblib"))
    posts = pd.read_pickle(os.path.join(MODELDIR, "posts.pkl"))
    try:
        with open(os.path.join(MODELDIR, "metrics.json")) as fh:
            metrics = json.load(fh)
    except FileNotFoundError:
        metrics = {}
    return img, txt, posts, metrics


@st.cache_resource(show_spinner=False)
def load_hashes():
    """md5 -> [filenames] for the labelled images, so a renamed copy of a
    dataset photo is still recognised.

    A list, not a single name: one photo in this dataset is filed three times
    under three names, twice as class 0 and once as class 1, so a digest can
    legitimately point at more than one post.
    """
    data_dir = L.find_data_dir()
    out = {}
    for lbl in (0, 1):
        folder = os.path.join(data_dir, "images", str(lbl))
        if not os.path.isdir(folder):
            continue
        for fn in os.listdir(folder):
            path = os.path.join(folder, fn)
            try:
                with open(path, "rb") as fh:
                    out.setdefault(hashlib.md5(fh.read()).hexdigest(), []).append(fn)
            except OSError:
                pass
    return out


def find_post(posts, filename, digest, hashes):
    """Match an upload to a dataset post by filename, then by content hash.

    Returns the row *label*, not the row: slicing with ``posts.loc[[idx]]``
    keeps the per-column dtypes, whereas ``.iloc[0].to_frame().T`` collapses
    every column to object and breaks the numeric feature maths downstream.
    """
    hit = posts.index[posts["image_file"] == filename]
    if len(hit):
        return hit[0], "filename"
    for fn in hashes.get(digest, []):
        hit = posts.index[posts["image_file"] == fn]
        if len(hit):
            return hit[0], "content hash"
    return None, None


def verdict_card(prob, source, note):
    yes = prob >= 0.5
    conf = prob if yes else 1 - prob
    st.markdown(
        f"""<div class="verdict {'yes' if yes else 'no'}">
              <div class="src">{source}</div>
              <h2>{'🍛 Pav bhaji' if yes else '🚫 Not pav bhaji'}
                  &nbsp;<span style="opacity:.55;font-size:1.05rem">
                  {conf:.0%} confident</span></h2>
              <p>{note}</p>
            </div>""", unsafe_allow_html=True)
    st.progress(float(prob))
    st.caption(f"p(pav bhaji) = {prob:.3f} &nbsp;·&nbsp; threshold 0.50")


# ------------------------------------------------------------------ page
st.title("🍛 Pav Bhaji Classifier")
st.caption("Upload a food photo and the model decides whether it is pav bhaji.")

bundle = load_models()
if bundle is None:
    st.error("Models not found. Train them first:\n\n"
             "```\npython app/train_models.py\n```")
    st.stop()

img_model, txt_model, posts, metrics = bundle

with st.sidebar:
    st.subheader("Models")
    if metrics.get("image"):
        m = metrics["image"]
        st.markdown("**Image model** — reads pixels")
        st.caption(f"5-fold CV accuracy **{m['accuracy']:.1%}**, "
                   f"ROC-AUC {m['roc_auc']:.3f}, trained on 452 photos")
    if metrics.get("text"):
        m = metrics["text"]
        st.markdown("**Caption model** — the notebook's model")
        st.caption(f"5-fold CV accuracy **{m['accuracy']:.1%}**, "
                   f"ROC-AUC {m['roc_auc']:.3f}, reads no pixels at all")
    st.caption("Majority-class baseline: 59.5%")
    st.divider()
    st.caption("452 labelled posts · 183 pav bhaji / 269 not. "
               "Both numbers are cross-validated, not training-set scores.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"],
                            label_visibility="collapsed")

if uploaded is None:
    st.info("Drop in a `.jpg` or `.png`. Images from `dataset/images/` are "
            "recognised automatically and also get the caption model's verdict.")
    st.stop()

data = uploaded.getvalue()
digest = hashlib.md5(data).hexdigest()
image = Image.open(io.BytesIO(data)).convert("RGB")

left, right = st.columns([1, 1.15])
with left:
    st.image(image, use_container_width=True)
    st.caption(f"`{uploaded.name}` · {image.width}x{image.height}")

# ---- image model -------------------------------------------------------
with st.spinner("Reading the image..."):
    feats = extract_features(image).reshape(1, -1)
    img_prob = float(img_model.predict_proba(feats)[0, 1])

with right:
    verdict_card(img_prob, "Image model · pixels",
                 "Colour geometry and texture of the photo itself.")

# ---- caption model (dataset images only) --------------------------------
idx, how = find_post(posts, uploaded.name, digest, load_hashes())

if idx is not None:
    row = posts.loc[[idx]]
    post = posts.loc[idx]
    feat_row = L.build_features(row.assign(
        caption=row["caption"].fillna(""),
        tags=row["tags"].map(lambda t: t if isinstance(t, list) else [])))
    txt_prob = float(txt_model.predict_proba(feat_row)[0, 1])

    st.divider()
    st.markdown(f"**Recognised as a dataset post** (matched by {how}) — so the "
                "notebook's caption-only model can weigh in too.")
    verdict_card(txt_prob, "Caption model · text + metadata",
                 "Caption, hashtags and post metadata. No pixels.")

    if pd.notna(post["label"]):
        truth = int(post["label"])
        st.markdown(f"**Ground truth:** {'🍛 pav bhaji' if truth else '🚫 not pav bhaji'}")
        cols = st.columns(2)
        cols[0].metric("Image model",
                       "correct" if (img_prob >= .5) == bool(truth) else "wrong")
        cols[1].metric("Caption model",
                       "correct" if (txt_prob >= .5) == bool(truth) else "wrong")

    with st.expander("Why the caption model said that"):
        f = feat_row.iloc[0]
        caption = (post["caption"] or "").strip()
        st.text_area("Caption", caption[:1500] or "(empty)", height=140,
                     disabled=True)
        evidence = pd.DataFrame([
            ("Pav bhaji named in the prose", "yes" if f["pb_in_body"] else "no"),
            ("Pav bhaji in the first 120 chars", "yes" if f["pb_in_lead"] else "no"),
            ("First dish named in the caption",
             {1.0: "pav bhaji", 0.0: "a rival dish", 0.5: "none"}[f["first_dish_caption"]]),
            ("Rival dish in the prose", "yes" if f["comp_in_body"] else "no"),
            ("Distinct rival dishes tagged", int(f["comp_tag_count"])),
            ("Hashtags on the post", int(f["n_tags"])),
            ("Prose length (chars)", int(f["body_len"])),
        ], columns=["signal", "value"])
        st.dataframe(evidence, hide_index=True, use_container_width=True)
        tags = [t for t in (post["tags"] or [])][:30]
        if tags:
            st.caption("tags: " + " ".join("#" + str(t) for t in tags))
else:
    st.divider()
    st.caption("Not a dataset image, so only the pixel model can judge it — the "
               "notebook's model needs a caption and hashtags, which an uploaded "
               "file does not carry.")
