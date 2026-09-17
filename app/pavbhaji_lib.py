"""Shared pav-bhaji logic: dataset loading, caption feature engineering and the
text pipeline. Lifted from satyajeet_yadav.ipynb so the app and the notebook can
never disagree about what the model sees.
"""
import json
import os
import re
import unicodedata
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, TransformerMixin

RANDOM_STATE = 42
LEAD_CHARS = 120

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def find_data_dir():
    for p in ["dataset", "../dataset", os.path.join(ROOT, "dataset")]:
        if os.path.exists(os.path.join(p, "pavbhaji.json")):
            return p
    raise FileNotFoundError("pavbhaji.json not found")


# ---------------------------------------------------------------- lexicons
HASHTAG_RE = re.compile(r"#([^\s#@]+)")
MENTION_RE = re.compile(r"@([A-Za-z0-9._]+)")

PAVBHAJI_RE = re.compile("|".join([
    r"p[ae]*[vo]+bh?[aeu]*j+i+",
    r"bh?[a]+j+ip[ae]*[vo]",
    r"masalapav",
]))

PAVBHAJI_SUPPORT = ["bhaji", "bhajji", "ladipav", "amulbutter", "butterbhaji",
                    "chowpatty", "pavbhajifondue"]

COMPETITOR_DISHES = [
    "vadapav", "panipuri", "golgappa", "golgappe", "puchka", "phuchka",
    "bhelpuri", "bhel", "dahipuri", "sevpuri", "dahibhalla", "alootikki",
    "tikki", "rajkachori", "kachori", "samosa", "chaat", "papdichaat",
    "frankie", "misal", "misalpav", "dabeli", "sandwich", "grilledsandwich",
    "dosa", "masaladosa", "idli", "uttapam", "utappam", "sambar", "rasam",
    "appam", "pongal", "medhuvada", "vada",
    "cholebhature", "bhature", "chole", "poori", "puri", "rajma", "kadhi",
    "paneertikka", "paneerbuttermasala", "shahipaneer", "dalmakhani",
    "butterchicken", "chickentikka", "tandoorichicken", "kebab", "seekhkebab",
    "biryani", "pulao", "khichdi", "naan", "kulcha", "paratha", "roti",
    "thali", "curry",
    "poha", "upma", "sabudana", "dhokla", "khandvi", "thepla", "pakoda",
    "pakora", "breadpakoda", "maggi", "maggie", "omelette", "bhurji",
    "eggroll", "momo", "momos", "springroll", "noodles", "chowmein",
    "manchurian", "friedrice", "hakkanoodles",
    "pizza", "burger", "pasta", "fries", "frenchfries", "garlicbread",
    "shawarma", "sushi", "taco", "nachos", "hotdog",
    "jalebi", "gulabjamun", "rasmalai", "rasgulla", "halwa", "barfi",
    "laddu", "kulfi", "falooda", "icecream", "cake", "brownie", "waffle",
    "pancake", "donut", "cupcake", "croissant", "pastry", "chocolate",
    "milkshake", "lassi", "chai", "coffee", "mojito",
    "chicken", "mutton", "fish", "prawn", "egg",
]
COMPETITOR_RE = re.compile("|".join(sorted(COMPETITOR_DISHES, key=len, reverse=True)))

SPAM_TAGS = {"like4like", "likeforlike", "likeforlikes", "l4l", "f4f", "follow",
             "followme", "follow4follow", "instagood", "instadaily",
             "picoftheday", "photooftheday", "tflers", "20likes", "instalike",
             "bestoftheday", "instafollow", "likes", "tagsforlikes", "igers"}

EMOJI_RE = re.compile("[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF"
                      "\U0001F1E6-\U0001F1FF" "\U00002B00-\U00002BFF"
                      "\U0000FE0F" "]")


# ---------------------------------------------------------------- helpers
def basename(url):
    return os.path.basename((url or "").split("?")[0])


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c))


def squash(s):
    return re.sub(r"[^a-z0-9]+", "", strip_accents(s).lower())


def tokenise(s):
    s = strip_accents(s)
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    s = re.sub(r"[^A-Za-z0-9]+", " ", s).lower()
    return re.sub(r"\s+", " ", s).strip()


def first_match_pos(pattern, sq):
    if not sq:
        return 1.0
    m = pattern.search(sq)
    return (m.start() / len(sq)) if m else 1.0


def first_dish(sq):
    pb, cp = PAVBHAJI_RE.search(sq), COMPETITOR_RE.search(sq)
    if pb and cp:
        return 1.0 if pb.start() <= cp.start() else 0.0
    return 1.0 if pb else (0.0 if cp else 0.5)


def split_caption(caption):
    caption = caption or ""
    hashtags = [h.lower() for h in HASHTAG_RE.findall(caption)]
    mentions = [m.lower() for m in MENTION_RE.findall(caption)]
    body = MENTION_RE.sub(" ", HASHTAG_RE.sub(" ", caption))
    body = re.sub(r"^[\s.•➖〰~\-_=+*]+$", " ", body, flags=re.M)
    return re.sub(r"\s+", " ", body).strip(), hashtags, mentions


class DenseTransformer(BaseEstimator, TransformerMixin):
    """Sparse -> dense, so tree ensembles can consume the matrix."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if issparse(X) else np.asarray(X)


# ---------------------------------------------------------------- loading
def load_dataset(data_dir):
    with open(os.path.join(data_dir, "pavbhaji.json"), encoding="utf-8") as fh:
        posts = json.load(fh)

    label_by_file = {}
    for lbl in (0, 1):
        folder = os.path.join(data_dir, "images", str(lbl))
        if os.path.isdir(folder):
            for fn in os.listdir(folder):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    label_by_file[fn] = lbl

    records, seen = [], set()
    for p in posts:
        pid = p.get("id")
        if pid in seen:
            continue
        seen.add(pid)

        fname = basename(p.get("display_url"))
        if fname not in label_by_file:
            fname = basename(p.get("thumbnail_src"))

        edges = (p.get("edge_media_to_caption") or {}).get("edges") or []
        dims = p.get("dimensions") or {}
        loc = p.get("location") or {}

        records.append(dict(
            id=pid,
            shortcode=p.get("shortcode"),
            image_file=fname,
            label=label_by_file.get(fname, np.nan),
            caption=edges[0]["node"]["text"] if edges else "",
            tags=p.get("tags") or [],
            likes=(p.get("edge_liked_by") or {}).get("count", 0) or 0,
            comments=(p.get("edge_media_to_comment") or {}).get("count", 0) or 0,
            is_video=bool(p.get("is_video")),
            comments_disabled=bool(p.get("comments_disabled")),
            owner_id=(p.get("owner") or {}).get("id"),
            width=dims.get("width") or 0,
            height=dims.get("height") or 0,
            timestamp=p.get("taken_at_timestamp") or 0,
            has_location=int(bool(loc)),
            location_name=(loc.get("name") if isinstance(loc, dict) else "") or "",
        ))
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------- features
def build_features(df):
    out = df.copy()

    parsed = out["caption"].fillna("").map(split_caption)
    out["body"] = [b for b, _, _ in parsed]
    out["caption_hashtags"] = [h for _, h, _ in parsed]
    out["mentions"] = [m for _, _, m in parsed]
    out["all_tags"] = [list(dict.fromkeys(list(ch) + [str(t).lower() for t in tg]))
                       for ch, tg in zip(out["caption_hashtags"], out["tags"])]

    out["body_text"] = out["body"].map(tokenise)
    out["tag_text"] = out["all_tags"].map(lambda ts: " ".join(tokenise(t) for t in ts))
    out["lead_text"] = out["caption"].fillna("").str[:LEAD_CHARS].map(tokenise)

    sq_body = out["body"].map(squash)
    sq_lead = out["caption"].fillna("").str[:LEAD_CHARS].map(squash)
    sq_cap = out["caption"].fillna("").map(squash)

    out["pb_in_body"] = sq_body.map(lambda s: int(bool(PAVBHAJI_RE.search(s))))
    out["pb_in_lead"] = sq_lead.map(lambda s: int(bool(PAVBHAJI_RE.search(s))))
    out["pb_in_tags"] = out["all_tags"].map(
        lambda ts: int(any(PAVBHAJI_RE.search(squash(t)) for t in ts)))
    out["pb_body_count"] = sq_body.map(lambda s: len(PAVBHAJI_RE.findall(s)))
    out["pb_caption_count"] = sq_cap.map(lambda s: len(PAVBHAJI_RE.findall(s)))
    out["pb_first_pos"] = sq_cap.map(lambda s: first_match_pos(PAVBHAJI_RE, s))
    out["pb_support"] = sq_cap.map(lambda s: int(any(w in s for w in PAVBHAJI_SUPPORT)))
    out["pb_tag_rank"] = out["caption_hashtags"].map(
        lambda ts: next((i / max(len(ts), 1) for i, t in enumerate(ts)
                         if PAVBHAJI_RE.search(squash(t))), 1.0))
    out["pb_tag_top3"] = out["caption_hashtags"].map(
        lambda ts: int(any(PAVBHAJI_RE.search(squash(t)) for t in ts[:3])))

    out["comp_in_body"] = sq_body.map(lambda s: int(bool(COMPETITOR_RE.search(s))))
    out["comp_in_lead"] = sq_lead.map(lambda s: int(bool(COMPETITOR_RE.search(s))))
    out["comp_body_count"] = sq_body.map(lambda s: len(set(COMPETITOR_RE.findall(s))))
    out["comp_tag_count"] = out["all_tags"].map(
        lambda ts: len({d for t in ts for d in COMPETITOR_RE.findall(squash(t))}))
    out["comp_tag_top3"] = out["caption_hashtags"].map(
        lambda ts: int(any(COMPETITOR_RE.search(squash(t)) for t in ts[:3])))
    out["dish_purity_body"] = out["pb_body_count"] / (
        out["pb_body_count"] + out["comp_body_count"] + 1e-6)
    out["dish_purity_tags"] = out["pb_in_tags"] / (
        out["pb_in_tags"] + out["comp_tag_count"] + 1e-6)
    out["dish_tag_density"] = out["comp_tag_count"] / out["all_tags"].map(
        lambda ts: max(len(ts), 1))
    out["first_dish_body"] = sq_body.map(first_dish)
    out["first_dish_lead"] = sq_lead.map(first_dish)
    out["first_dish_caption"] = sq_cap.map(first_dish)
    out["loc_is_pb"] = out["location_name"].map(
        lambda s: int(bool(PAVBHAJI_RE.search(squash(s)))))

    cap = out["caption"].fillna("")
    out["n_tags"] = out["all_tags"].map(len)
    out["n_caption_hashtags"] = out["caption_hashtags"].map(len)
    out["n_mentions"] = out["mentions"].map(len)
    out["n_spam_tags"] = out["all_tags"].map(lambda ts: sum(t in SPAM_TAGS for t in ts))
    out["caption_len"] = cap.str.len()
    out["body_len"] = out["body"].str.len()
    out["body_words"] = out["body_text"].str.split().map(len)
    out["hashtag_char_ratio"] = cap.map(
        lambda c: sum(len(h) + 1 for h in HASHTAG_RE.findall(c)) / max(len(c), 1))
    out["body_char_ratio"] = out["body_len"] / out["caption_len"].clip(lower=1)
    out["n_emoji"] = cap.map(lambda c: len(EMOJI_RE.findall(c)))
    out["n_newlines"] = cap.str.count("\n")
    out["n_dot_lines"] = cap.map(
        lambda c: len(re.findall(r"^\s*[.•➖]\s*$", c, flags=re.M)))
    out["n_follow"] = cap.str.lower().str.count("follow")
    out["is_repost"] = cap.str.lower().str.contains(
        r"repost|regrann|get_repost|pc\s*@|pic credit").astype(int)
    out["upper_ratio"] = cap.map(
        lambda c: sum(ch.isupper() for ch in c) / max(sum(ch.isalpha() for ch in c), 1))

    out["log_likes"] = np.log1p(out["likes"])
    out["log_comments"] = np.log1p(out["comments"])
    out["comment_rate"] = out["comments"] / out["likes"].clip(lower=1)
    out["aspect"] = out["width"] / out["height"].clip(lower=1)
    out["is_square"] = (np.abs(out["aspect"] - 1.0) < 0.02).astype(int)
    out["is_portrait"] = (out["aspect"] < 0.95).astype(int)
    out["megapixels"] = out["width"] * out["height"] / 1e6
    dt = out["timestamp"].map(
        lambda t: datetime.fromtimestamp(int(t), tz=timezone.utc) if t else None)
    out["hour"] = dt.map(lambda d: d.hour if d else 12)
    out["dow"] = dt.map(lambda d: d.weekday() if d else 3)
    out["month"] = dt.map(lambda d: d.month if d else 6)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    return out


NUMERIC_FEATURES = [
    "pb_in_body", "pb_in_lead", "pb_in_tags", "pb_body_count", "pb_caption_count",
    "pb_first_pos", "pb_tag_rank", "pb_tag_top3", "pb_support",
    "comp_in_body", "comp_in_lead", "comp_body_count", "comp_tag_count",
    "comp_tag_top3", "dish_purity_body", "dish_purity_tags", "dish_tag_density",
    "first_dish_body", "first_dish_lead", "first_dish_caption", "loc_is_pb",
    "n_tags", "n_caption_hashtags", "n_mentions", "n_spam_tags", "caption_len",
    "body_len", "body_words", "hashtag_char_ratio", "body_char_ratio",
    "n_emoji", "n_newlines", "n_dot_lines", "n_follow", "is_repost", "upper_ratio",
    "log_likes", "log_comments", "comment_rate", "aspect", "is_square",
    "is_portrait", "megapixels", "has_location", "hour_sin", "hour_cos",
    "dow", "month",
]


# ---------------------------------------------------------------- pipeline
def feature_union(min_df=2):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer

    return ColumnTransformer([
        ("body", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df,
                                 sublinear_tf=True, strip_accents="unicode",
                                 max_features=20000), "body_text"),
        ("tags", TfidfVectorizer(min_df=min_df, sublinear_tf=True,
                                 strip_accents="unicode",
                                 max_features=10000), "tag_text"),
        ("lead", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=min_df, sublinear_tf=True,
                                 max_features=20000), "lead_text"),
        ("meta", StandardScaler(), NUMERIC_FEATURES),
    ], remainder="drop", sparse_threshold=0.3)


def text_model():
    """The notebook's winning model: soft-vote of logistic regression + extra trees."""
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier

    lr = Pipeline([("feats", feature_union()),
                   ("clf", LogisticRegression(C=0.5, max_iter=4000,
                                              class_weight="balanced",
                                              random_state=RANDOM_STATE))])
    et = Pipeline([("feats", feature_union(min_df=3)),
                   ("dense", DenseTransformer()),
                   ("clf", ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2,
                                                class_weight="balanced", n_jobs=-1,
                                                random_state=RANDOM_STATE))])
    return VotingClassifier([("lr", lr), ("et", et)], voting="soft")
