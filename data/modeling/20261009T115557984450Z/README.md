# One table, two targets

Use model_dataset.csv for both regression and classification. Both targets and all future-window metadata are excluded from feature_columns in targets_metadata.json. Select the desired target as y; never include the other target in X. Use dataset_split to select train, validation and test rows. Exclude purged and unlabeled rows. Null targets mean insufficient complete future returns and are not zero risk.

Risk is annualized next-20-session realized volatility, in decimal units. Classes use training-only pooled tertiles with fixed thresholds. SPY is a benchmark feature, not a modeled company. Targets use future data intentionally; features must not. Rows assume prediction after the current close. The source's unresolved corporate-action and historical identity checks remain applicable.
