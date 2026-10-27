# Crop Recommendation Dataset (Atharva Ingle)

- **Role:** PRIMARY - used by all training code
- **Landing page:** https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
- **Reference notebook:** https://www.kaggle.com/code/atharvaingle/what-crop-to-grow
- **License:** Kaggle page does not state a clear license (author says it was assembled from public India rainfall/climate/fertilizer data). A CC BY 4.0 copy exists on Figshare.
- **Rows:** 2,200
- **Columns:** N, P, K, temperature, humidity, ph, rainfall, label
- **Classes / target:** 22 crops x 100 rows (perfectly balanced)
- **Download:** `https://www.kaggle.com/api/v1/datasets/download/atharvaingle/crop-recommendation-dataset` (zip, no login needed when this was written); fallback `https://ndownloader.figshare.com/files/47707906`
- **Expected sha256:** `54a5a6e5408668e668667efc50de2fc867c1b875e0431b4f54dd331b0a109a4e`

Kaggle's public download endpoint worked WITHOUT login (HTTP 200, 65 KB zip). Verified byte-identical to the Figshare CC BY 4.0 copy (https://figshare.com/articles/dataset/Crop_Recommendation_dataset/26308696); a Hugging Face Space copy has identical values (only line endings differ). Clean, balanced, no nulls: published papers report ~98-99.6% accuracy.

_Generated from datasets/manifest.json by `fetch_datasets.py --docs`. Re-fetch: `python3 datasets/fetch_datasets.py --only crop_recommendation_kaggle_atharvaingle`._
