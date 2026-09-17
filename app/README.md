# Pav Bhaji Classifier — upload UI

A small Streamlit app that takes an image and says whether it is pav bhaji.

## Run it

```bat
run_app.bat
```

or manually:

```bash
python app/train_models.py     # once — writes models/
python -m streamlit run app/app.py
```

It opens at <http://localhost:8501>.

## Two models, because the notebook's model is text-only

`sarthik_khanna.ipynb` builds a classifier that deliberately never looks at a
pixel — it reads the Instagram caption, hashtags and metadata. That model cannot
score a bare uploaded photo, because a photo carries no caption. So the app
serves two:

| model | input | works on |
|---|---|---|
| **image model** | pixels — HSV histograms, a 3×3 colour grid, centre-vs-border contrast, warm-saturated colour mass, edge density | any uploaded image |
| **caption model** | the notebook's soft-vote (logistic regression + extra trees) over caption/hashtag/metadata features | images recognised as dataset posts |

When you upload one of the 452 labelled photos (or any of the 1,495 posts'
images), the app matches it by filename — falling back to an MD5 of the file
contents, so a renamed copy still resolves — looks up that post's caption, and
shows both verdicts side by side along with the ground-truth label.

For anything else, only the pixel model speaks, and the app says so.

## Files

- `pavbhaji_lib.py` — dataset loading + the notebook's caption feature engineering, shared so app and notebook cannot drift
- `image_model.py` — the pixel descriptor (PIL + numpy; no torchvision in this environment) and its classifier
- `train_models.py` — cross-validates both models, then refits on everything and saves to `models/`
- `app.py` — the Streamlit UI

## Honesty note

The accuracies in the sidebar are 5-fold cross-validated, not training-set
scores. With 452 labelled images the error bars are roughly ±4 points. The
classical colour/texture descriptor is a reasonable baseline, not a fine-tuned
CNN — `torchvision` is not installed in this environment, so pretrained
ImageNet features were not an option.
