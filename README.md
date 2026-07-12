# MIA5126 Masonry Value Estimator

This folder is ready to upload to a public GitHub repository and deploy on Streamlit Community Cloud.

## Files Included

- `app.py` - Streamlit web app
- `requirements.txt` - Python dependencies for Streamlit Cloud
- `masonry_values_filled.csv` - final enriched dataset with reported + model-estimated masonry values. The app uses this file for both dataset views.
- `.streamlit/config.toml` - app theme settings

## Deploy

1. Create a public GitHub repository.
2. Upload the contents of this folder.
3. Go to Streamlit Community Cloud.
4. Create a new app from the GitHub repository.
5. Set the main file path to:

```text
app.py
```

The app uses `masonry_values_filled.csv` from this same folder.
