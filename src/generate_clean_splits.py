import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

SPLITS_DIR = Path("data/splits_combined")

# 1. Load the current overlapping splits
train_df = pd.read_csv(SPLITS_DIR / "train.csv")
val_df = pd.read_csv(SPLITS_DIR / "val.csv")
test_df = pd.read_csv(SPLITS_DIR / "test.csv")

# 2. Merge everything into a single Master Pool
master_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
print(f"Total rows before deduplication: {len(master_df)}")

# 3. Standardize text for strict deduplication (lowercase, strip whitespace)
master_df['clean_key'] = master_df['Patient Question'].str.strip().str.lower()
master_clean = master_df.drop_duplicates(subset=['clean_key'], keep='first').copy()
master_clean = master_clean.drop(columns=['clean_key'])
print(f"Total rows after deduplication: {len(master_clean)}")

# 4. Perform the Stratified Splits (80% Train, 10% Val, 10% Test)
# First split: 80% train, 20% temp (val + test)
train_clean, temp = train_test_split(
    master_clean, 
    test_size=0.20, 
    random_state=42, 
    stratify=master_clean['y_mc']
)

# Second split: split the 20% temp perfectly in half (10% val, 10% test)
val_clean, test_clean = train_test_split(
    temp, 
    test_size=0.50, 
    random_state=42, 
    stratify=temp['y_mc']
)

# 5. Overwrite the old leaky files with the clean splits
train_clean.to_csv(SPLITS_DIR / "train.csv", index=False)
val_clean.to_csv(SPLITS_DIR / "val.csv", index=False)
test_clean.to_csv(SPLITS_DIR / "test.csv", index=False)

print(f"\n✅ Clean splits generated successfully!")
print(f"Train: {len(train_clean)} | Val: {len(val_clean)} | Test: {len(test_clean)}")