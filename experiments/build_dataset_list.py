"""
Curated 60-dataset list for PW-NB experiment.
Famous, benchmark-quality datasets covering 5 experimental buckets.

Design:
  high_dim   (12) -- >100 features             -- tests PW-NB on high-dimensional data
  imbalanced (14) -- IR > 5                    -- PW-NB core strength
  many_class (10) -- >=8 balanced classes      -- multi-class coverage
  large_n    ( 8) -- >10K instances, IR<=5     -- scalability
  standard   (16) -- general UCI/OpenML benchmarks

Cluster contamination removed vs naive random sampling:
  - NASA defects: kept only kc1 (1067) + mc1 (1056); dropped pc1/pc3/pc4/mw1/jm1
  - mfeat suite:  kept only mfeat-factors (12) + mfeat-karhunen (16); dropped pixel/zernike/fourier/morphological
  - BNG(breast-w): synthetic blowup of breast-w, dropped

Run ONCE, commit datasets_selected.csv to git, never rerun.
Usage:
    python experiments/build_dataset_list.py
"""

import openml
import pandas as pd

# Curated list: (OpenML DID, canonical_name, bucket)
# DIDs are pinned to avoid name-matching ambiguity.
CURATED = [
    # HIGH DIM (12): >100 features
    (1501,   "semeion",       "high_dim"),   # 1593x257,  10c -- handwritten digits
    (12,     "mfeat-factors", "high_dim"),   # 2000x217,  10c -- digit features (1 mfeat kept)
    (41082,  "USPS",           "high_dim"),   # 9298x257,  10c -- USPS handwritten digits (replaces mnist_784: kNN cost infeasible at 70K×785)
    (1485,   "madelon",       "high_dim"),   # 2600x501,   2c -- NIPS 2003 feature selection
    (40665,  "clean1",        "high_dim"),   # 476x169,    2c -- EEG
    (40666,  "clean2",        "high_dim"),   # 6598x169,   2c -- EEG
    (1038,   "gina_agnostic", "high_dim"),   # 3468x971,   2c -- NIPS 2003
    (300,    "isolet",        "high_dim"),   # 7797x618,  26c -- spoken letter recognition
    (312,    "scene",         "high_dim"),   # 2407x300,   2c -- image scene features
    (1478,   "har",           "high_dim"),   # 10299x562,  6c -- human activity recognition
    (40910,  "Speech",        "high_dim"),   # 3686x401,   2c -- speech features
    (1476,   "gas-drift",     "high_dim"),   # 13910x129,  6c -- sensor drift (replaces mfeat-pixel)

    # IMBALANCED (14): IR > 5, diverse domains
    (30,     "page-blocks",                      "imbalanced"),  # 5473x11,  5c, IR=176
    (1067,   "kc1",                              "imbalanced"),  # 2109x22,  2c, IR=5.5  -- software defects
    (1056,   "mc1",                              "imbalanced"),  # 9466x39,  2c, IR=138  -- software defects (different scale)
    (39,     "ecoli",                             "imbalanced"),  # 336x8,    8c, IR=72   -- same domain as yeast, no format issues
    (41,     "glass",                            "imbalanced"),  # 214x10,   6c, IR=8.4
    (40497,  "thyroid-ann",                      "imbalanced"),  # 3772x22,  3c, IR=38   -- medical
    (40983,  "wilt",                             "imbalanced"),  # 4839x6,   2c, IR=18   -- remote sensing
    (1487,   "ozone-level-8hr",                  "imbalanced"),  # 2534x73,  2c, IR=15   -- environmental
    (1467,   "climate-model-simulation-crashes", "imbalanced"),  # 540x21,   2c, IR=11   -- simulation
    (1466,   "cardiotocography",                 "imbalanced"),  # 2126x36, 10c, IR=11   -- medical
    (40691,  "wine-quality-red",                 "imbalanced"),  # 1599x12,  6c, IR=68   -- food quality
    (40498,  "wine-quality-white",               "imbalanced"),  # 4898x12,  7c, IR=440  -- food quality
    (46936,  "internet_firewall",                "imbalanced"),  # 65532x12, 4c, IR=697  -- network security
    (44232,  "UCI_churn",                        "imbalanced"),  # 3333x21,  2c, IR=5.9  -- telecom

    # MANY CLASS (10): >=8 balanced classes, diverse domains
    (28,     "optdigits",            "many_class"),  # 5620x65,  10c -- optical digits
    (32,     "pendigits",            "many_class"),  # 10992x17, 10c -- pen digits
    (6,      "letter",               "many_class"),  # 20000x17, 26c -- letter recognition
    (307,    "vowel",                "many_class"),  # 990x13,   11c -- English vowel recognition
    (40499,  "texture",              "many_class"),  # 5500x41,  11c -- image texture
    (16,     "mfeat-karhunen",       "many_class"),  # 2000x65,  10c -- Karhunen-Loeve (1 mfeat kept)
    (1459,   "artificial-characters","many_class"),  # 10218x8,  10c
    (375,    "JapaneseVowels",       "many_class"),  # 9961x15,   9c -- speaker ID from Japanese vowels
    (40496,  "LED-display-domain-7digit", "many_class"),  # 500x8,  10c -- LED digit simulation (replaces diggle_table_a2: n/L=34 too small for stable CV)
    (46372,  "Multiclass_Classification_for_Corporate_Credit_Ratings", "many_class"),  # 5000x8, 10c -- finance

    # LARGE N (8): >10K instances, IR<=5
    (1120,   "MagicTelescope",                 "large_n"),  # 19020x12, 2c -- MAGIC telescope
    (1471,   "eeg-eye-state",                  "large_n"),  # 14980x15, 2c -- EEG eye state
    (151,    "electricity",                    "large_n"),  # 45312x9,  2c -- electricity pricing
    (4534,   "PhishingWebsites",               "large_n"),  # 11055x31, 2c -- phishing detection
    (42477,  "default-of-credit-card-clients", "large_n"),  # 30000x24, 2c -- credit default
    (1046,   "mozilla4",                       "large_n"),  # 15545x6,  2c -- Mozilla
    (43979,  "california",                     "large_n"),  # 20634x9,  2c -- California housing
    (1461,   "bank-marketing",                  "large_n"),  # 45211x17, 2c -- telecom marketing (replaces Run_or_walk: kNN cost at 88K)

    # STANDARD (16): classic UCI/OpenML benchmarks
    (61,     "iris",                             "standard"),  # 150x5,    3c
    (187,    "wine",                             "standard"),  # 178x14,   3c
    (15,     "breast-w",                         "standard"),  # 699x10,   2c -- 16 MV, handled by preprocessing
    (37,     "diabetes",                         "standard"),  # 768x9,    2c
    (40,     "sonar",                            "standard"),  # 208x61,   2c
    (59,     "ionosphere",                       "standard"),  # 351x35,   2c
    (54,     "vehicle",                          "standard"),  # 846x19,   4c
    (53,     "heart-statlog",                    "standard"),  # 270x14,   2c
    (182,    "satimage",                         "standard"),  # 6430x37,  6c
    (36,     "segment",                          "standard"),  # 2310x20,  7c
    (60,     "waveform-5000",                    "standard"),  # 5000x41,  3c
    (1462,   "banknote-authentication",          "standard"),  # 1372x5,   2c
    (1488,   "parkinsons",                       "standard"),  # 195x23,   2c
    (1464,   "blood-transfusion-service-center", "standard"),  # 748x5,    2c
    (1496,   "ringnorm",                         "standard"),  # 7400x21,  2c
    (44,     "spambase",                         "standard"),  # 4601x58,  2c
]


def main():
    print("Loading OpenML dataset catalog...")
    all_ds = openml.datasets.list_datasets(output_format="dataframe")

    rows = []
    issues = []

    bucket_counts = {}
    for _, _, b in CURATED:
        bucket_counts[b] = bucket_counts.get(b, 0) + 1
    print(f"Target distribution: {bucket_counts}  (total={sum(bucket_counts.values())})")

    print(f"\n{'DID':>6}  {'Name':<45}  {'n':>7}  {'f':>4}  {'c':>3}  {'IR':>7}  {'bucket':<12}  notes")
    print("-" * 115)

    for did, name, bucket in CURATED:
        if did not in all_ds.index:
            print(f"{did:>6}  {name:<45}  NOT IN CATALOG")
            issues.append((did, name, "not_found"))
            continue

        row = all_ds.loc[did]
        n   = row["NumberOfInstances"]
        f   = row["NumberOfFeatures"]
        c   = row["NumberOfClasses"]
        mv  = row["NumberOfMissingValues"]
        sym = row["NumberOfSymbolicFeatures"]
        maj = row["MajorityClassSize"]
        mi  = row["MinorityClassSize"]
        ir  = maj / mi if (not pd.isna(mi) and mi > 0) else float("nan")

        notes = []
        if not pd.isna(mv) and mv > 0:
            notes.append(f"MV={mv:.0f}")
        if not pd.isna(sym) and sym > 1:
            notes.append(f"sym={sym:.0f}")
        note_str = ", ".join(notes) if notes else "OK"

        print(f"{did:>6}  {name:<45}  {n:>7.0f}  {f:>4.0f}  {c:>3.0f}  {ir:>7.1f}  {bucket:<12}  {note_str}")

        rows.append({
            "did":                did,
            "name":               row["name"],
            "NumberOfInstances":  int(n)   if not pd.isna(n) else None,
            "NumberOfFeatures":   int(f)   if not pd.isna(f) else None,
            "NumberOfClasses":    int(c)   if not pd.isna(c) else None,
            "bucket":             bucket,
        })

    df = pd.DataFrame(rows)
    out = "datasets_selected.csv"
    df.to_csv(out, index=False)

    print(f"\n{'=' * 50}")
    print(f"Total: {len(df)} datasets saved to {out}")
    if issues:
        print(f"Issues: {issues}")

    print("\nBucket summary:")
    for b, grp in df.groupby("bucket"):
        print(f"  {b:<15} {len(grp):>2} datasets")


if __name__ == "__main__":
    main()
