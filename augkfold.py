import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

def load_features_for_object(
    feature_root: str | Path,
    object_type: str,
    n_splits: int = 5,
    fold_index: int = 0,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads DINOv2 feature vectors for one object type, splits originals
    via StratifiedKFold, then adds augmented features to train only.

    Args:
        feature_root:  Root folder containing dino_features/
        object_type:   e.g. "bottle"
        n_splits:      Number of folds (default 5, use len(originals) for LOO)
        fold_index:    Which fold to use as test set (0 to n_splits-1)
        random_state:  Reproducibility seed

    Returns:
        X_train, y_train, X_test, y_test as numpy arrays
    """
    feature_root = Path(feature_root)
    object_dir   = feature_root / object_type

    if not object_dir.exists():
        raise FileNotFoundError(f"Object directory not found: {object_dir}")

    # ── 1. Collect originals and augmented separately ────────────────────────
    original_features = []
    original_labels   = []
    augmented_features = []
    augmented_labels   = []

    for class_dir in sorted(object_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        feat_files = sorted(class_dir.glob("feat_*.npy"))

        for feat_path in feat_files:
            label_path = class_dir / feat_path.name.replace("feat_", "label_")
            if not label_path.exists():
                print(f"  Warning: no label file for {feat_path.name}, skipping.")
                continue

            feature = np.load(feat_path)
            label   = np.load(label_path)
            # label files may be scalar arrays — flatten to string/int
            label   = label.item() if label.ndim == 0 else label[0]

            is_augmented = "_aug" in feat_path.stem

            if is_augmented:
                augmented_features.append(feature)
                augmented_labels.append(label)
            else:
                original_features.append(feature)
                original_labels.append(label)

    if len(original_features) == 0:
        raise ValueError(f"No original feature files found under {object_dir}")

    X_orig = np.stack(original_features)   # [n_originals, feature_dim]
    y_orig = np.array(original_labels)     # [n_originals]

    n_originals = len(X_orig)
    print(f"  {object_type}: {n_originals} originals | "
          f"{len(augmented_features)} augmented samples")

    # ── 2. Decide split strategy ─────────────────────────────────────────────
    # For very small datasets use Leave-One-Out (n_splits = n_originals)
    if n_originals < 20:
        actual_splits = n_originals
        print(f"  → n < 20: switching to Leave-One-Out ({actual_splits} folds)")
    else:
        actual_splits = n_splits

    if fold_index >= actual_splits:
        raise ValueError(f"fold_index {fold_index} out of range for "
                         f"{actual_splits} folds.")

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True,
                          random_state=random_state)
    splits = list(skf.split(X_orig, y_orig))
    train_idx, test_idx = splits[fold_index]

    # ── 3. Build test set — originals only ───────────────────────────────────
    X_test = X_orig[test_idx]
    y_test = y_orig[test_idx]

    # ── 4. Build train set — originals + their augmentations ─────────────────
    X_train_orig = X_orig[train_idx]
    y_train_orig = y_orig[train_idx]

    # Map: base filename stem → whether it belongs to train fold
    # original feat files are named feat_000.npy → stem "feat_000"
    all_feat_files   = []
    all_feat_labels  = []
    for class_dir in sorted(object_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for feat_path in sorted(class_dir.glob("feat_*.npy")):
            label_path = class_dir / feat_path.name.replace("feat_", "label_")
            if label_path.exists():
                all_feat_files.append(feat_path)
                all_feat_labels.append(class_dir.name)

    # Cleaner: rebuild original file list in same order as X_orig
    original_file_list = []
    for class_dir in sorted(object_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for feat_path in sorted(class_dir.glob("feat_*.npy")):
            if "_aug" not in feat_path.stem:
                label_path = class_dir / feat_path.name.replace("feat_", "label_")
                if label_path.exists():
                    original_file_list.append(feat_path)

    train_original_stems = {
        original_file_list[i].stem for i in train_idx  # e.g. "feat_000"
    }
    # Build set of (class_name, stem) pairs for train-fold originals
    train_original_keys = {
        (original_file_list[i].parent.name, original_file_list[i].stem)
        for i in train_idx
    }
    # Also build test keys — for explicit safety check
    test_original_keys = {
        (original_file_list[i].parent.name, original_file_list[i].stem)
        for i in test_idx
    }

    # Add augmented samples whose base original is in train fold
    aug_features_to_add = []
    aug_labels_to_add   = []

    # for class_dir in sorted(object_dir.iterdir()):
    #     if not class_dir.is_dir():
    #         continue
    #     for feat_path in sorted(class_dir.glob("feat_*_aug*.npy")):
    #         label_path = class_dir / feat_path.name.replace("feat_", "label_")
    #         if not label_path.exists():
    #             continue

    #         # "feat_000_aug1" → base stem "feat_000"
    #         base_stem = feat_path.stem.split("_aug")[0]  # "feat_000"
    #         stem_num = base_stem.split("_")[1]
    #         print(stem_num)
            
    #         # TODO: check that the test augmentation images are left out!!!
    #         if base_stem in train_original_stems:
    #             feature = np.load(feat_path)
    #             label   = np.load(label_path)
    #             label   = label.item() if label.ndim == 0 else label[0]
    #             aug_features_to_add.append(feature)
    #             aug_labels_to_add.append(label)

    for class_dir in sorted(object_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for feat_path in sorted(class_dir.glob("feat_*_aug*.npy")):
            label_path = class_dir / feat_path.name.replace("feat_", "label_")
            if not label_path.exists():
                continue

            base_stem = feat_path.stem.split("_aug")[0]  # "feat_000"
            class_name = class_dir.name                  # "broken_large"
            key = (class_name, base_stem)

            # Explicit double check: in train AND not in test
            if key in train_original_keys and key not in test_original_keys:
                feature = np.load(feat_path)
                label   = np.load(label_path)
                label   = label.item() if label.ndim == 0 else label[0]
                aug_features_to_add.append(feature)
                aug_labels_to_add.append(label)
            elif key in test_original_keys:
                pass  # silently skip — correct behavior

    if aug_features_to_add:
        X_aug = np.stack(aug_features_to_add)
        y_aug = np.array(aug_labels_to_add)
        X_train = np.concatenate([X_train_orig, X_aug], axis=0)
        y_train = np.concatenate([y_train_orig, y_aug], axis=0)
    else:
        X_train = X_train_orig
        y_train = y_train_orig

    # print(f"  Fold {fold_index}: "
    #       f"X_train {X_train.shape} | X_test {X_test.shape}")
    # print(f"  Train class distribution: "
    #       f"{dict(zip(*np.unique(y_train, return_counts=True)))}")
    # print(f"  Test class distribution:  "
    #       f"{dict(zip(*np.unique(y_test, return_counts=True)))}")

    return X_train, X_test, y_train, y_test



def load_feature_indices_for_object(
    feature_root: str | Path,
    object_type: str,
    n_splits: int = 5,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Loads ALL DINOv2 feature vectors for one object type (originals +
    augmented) into a single pool, then returns index arrays for every fold.

    The fold split is performed on originals only via StratifiedKFold.
    Augmented samples are assigned to train if — and only if — their base
    original is in the train fold.  Test indices always contain originals only.

    Args:
        feature_root:  Root folder containing dino_features/
        object_type:   e.g. "bottle"
        n_splits:      Number of folds (default 5, use len(originals) for LOO)
        random_state:  Reproducibility seed

    Returns:
        X_all  : np.ndarray  [N_total, feature_dim]  – full feature pool
        y_all  : np.ndarray  [N_total]               – full label pool
        folds  : list of (train_idx, test_idx) tuples, one per fold.
                 Each train_idx / test_idx is a np.ndarray of indices into X_all.
    """
    feature_root = Path(feature_root)
    object_dir   = feature_root / object_type

    if not object_dir.exists():
        raise FileNotFoundError(f"Object directory not found: {object_dir}")

    # ── 1. Load ALL samples into a flat list, track which are originals ──────
    all_features: list[np.ndarray] = []
    all_labels:   list             = []
    # For each entry: (class_name, base_stem) or None for augmented entries
    # None would be confusing — store base key for every sample instead
    all_base_keys: list[tuple[str, str] | None] = []
    is_augmented_flag: list[bool] = []

    # Keep a separate ordered list of (global_index, base_key) for originals
    # so we can pass indices into StratifiedKFold
    original_global_indices: list[int] = []
    original_base_keys:      list[tuple[str, str]] = []
    original_labels_for_skf: list = []

    for class_dir in sorted(object_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        for feat_path in sorted(class_dir.glob("feat_*.npy")):
            label_path = class_dir / feat_path.name.replace("feat_", "label_")
            if not label_path.exists():
                print(f"  Warning: no label file for {feat_path.name}, skipping.")
                continue

            feature = np.load(feat_path)
            label   = np.load(label_path)
            label   = label.item() if label.ndim == 0 else label[0]

            augmented = "_aug" in feat_path.stem
            base_stem = feat_path.stem.split("_aug")[0] if augmented else feat_path.stem
            base_key  = (class_dir.name, base_stem)

            global_idx = len(all_features)
            all_features.append(feature)
            all_labels.append(label)
            all_base_keys.append(base_key)
            is_augmented_flag.append(augmented)

            if not augmented:
                original_global_indices.append(global_idx)
                original_base_keys.append(base_key)
                original_labels_for_skf.append(label)

    if not original_global_indices:
        raise ValueError(f"No original feature files found under {object_dir}")

    n_originals  = len(original_global_indices)
    n_augmented  = sum(is_augmented_flag)
    print(f"  {object_type}: {n_originals} originals | {n_augmented} augmented samples")

    # ── 2. Stack everything into arrays ─────────────────────────────────────
    X_all = np.stack(all_features)        # [N_total, feature_dim]
    y_all = np.array(all_labels)          # [N_total]

    # Arrays used for StratifiedKFold — originals only
    orig_idx_arr   = np.array(original_global_indices)   # global positions
    orig_label_arr = np.array(original_labels_for_skf)

    # ── 3. Decide split strategy ─────────────────────────────────────────────
    if n_originals < 20:
        actual_splits = n_originals
        print(f"  → n < 20: switching to Leave-One-Out ({actual_splits} folds)")
    else:
        actual_splits = n_splits

    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True,
                          random_state=random_state)
    # split() operates on positions within orig_idx_arr (0..n_originals-1)
    splits = list(skf.split(orig_idx_arr, orig_label_arr))

    # ── 4. Build index arrays for every fold ─────────────────────────────────
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    for fold_index, (fold_train_pos, fold_test_pos) in enumerate(splits):

        # Translate positions → global indices in X_all
        test_global_idx   = orig_idx_arr[fold_test_pos]
        train_orig_global = orig_idx_arr[fold_train_pos]

        train_base_key_set = {original_base_keys[p] for p in fold_train_pos}
        test_base_key_set  = {original_base_keys[p] for p in fold_test_pos}

        # Assign augmented samples whose base original is in this train fold
        aug_train_global = [
            g_idx
            for g_idx, (base_key, is_aug) in enumerate(
                zip(all_base_keys, is_augmented_flag)
            )
            if is_aug
            and base_key in train_base_key_set
            and base_key not in test_base_key_set
        ]

        train_indices = np.concatenate(
            [train_orig_global, np.array(aug_train_global, dtype=int)]
        )
        test_indices = test_global_idx  # originals only

        print(f"  Fold {fold_index}: "
              f"train {len(train_indices)} "
              f"({len(train_orig_global)} orig + {len(aug_train_global)} aug) | "
              f"test {len(test_indices)} (originals only)")

        folds.append((train_indices, test_indices))

    return X_all, y_all, folds