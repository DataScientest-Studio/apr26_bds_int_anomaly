# load a DINOv2 Modell and extract features from the defect images
# use extracted features to train a SVM, Logistic Regression, Random Forest or KNN for the Classification step

import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score, f1_score

from skopt import BayesSearchCV
from skopt.space import Real

import pandas as pd

import augkfold as akf
from collections import defaultdict
import io
import json
import contextlib
import joblib


ALL_OBJECT_TYPES = [
    "bottle","cable","capsule","carpet","grid","hazelnut",
    "leather","metal_nut","pill","screw","tile","toothbrush",
    "transistor","wood","zipper"
]
# FEAT_OUTPUT_DIR = Path("./dino_feature_dataset")
OUTPUT_DIR = Path("./dino_features")

DATASET_ROOT_PATH = r"C:\Users\Tabea\Desktop\Liora_Weiterbildung\Projects\Group_Project\MVTec_AD\archive_aug"

# 1) get all the defect images of one object type and extract the feature vectors
root_path = DATASET_ROOT_PATH
root_path = root_path.replace("\\", "/")
root_path = Path(root_path)
DATA_DIR = root_path

OBJECT_TYPE = ["bottle"]

def extract_features(img_path, model, device):
    # load image, extract features from choosen modell and flatten
    # ImageNet tranformation
    transform = T.Compose([
        T.Resize((518, 518)), # multiple of 14
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad(): # unable gradient update
        features = model(tensor)
    return features.cpu().numpy().flatten()


def extract_extrafeatures(img_path, model, device):
    """Extrahiert ein Hybrid-Feature (Kombination aus CLS und gemittelten Patches).

    Erfasst sowohl grobe, globale Fehler als auch feine, lokale Defekte.
    """
    # ImageNet tranformation
    transform = T.Compose([
        T.Resize((518, 518)), # multiple of 14
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Holt die Features der allerletzten Transformer-Schicht (Index -1)
        # n_last_blocks=1 liefert eine Liste mit einem Tensor zurück
        layer_output = model.get_intermediate_layers(tensor, n=1, return_class_token=True)
        
        # DINOv2 liefert hier standardmäßig nur die Patch-Features (ohne CLS-Token)
        # Shape von layer_output: [Batch_Size, Anzahl_Patches, 1024]
        # Bei Auflösung 518x518 und Patch-Größe 14x14 gibt es genau 1369 Patches
        patch_tokens, cls_token = layer_output[0]

        # Mittelwert über alle Bild-Patches bilden (erfasst feine Defekte)
        patch_features = torch.mean(patch_tokens, dim=1)
        
        # 3. Beide Vektoren verketten (Concatenate)
        # Dadurch entsteht ein mächtiger Vektor der Größe 2048
        hybrid_feature = torch.cat((cls_token, patch_features), dim=1)
        
    return hybrid_feature.cpu().numpy().flatten()


def generate_DINOv2_features(object_type):
    object_dir = DATA_DIR / object_type / "test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device).eval()

    defect_class_names = []

    if os.path.exists(object_dir):
        dirs = sorted(os.listdir(object_dir))
        defect_dirs = [defect_class for defect_class in dirs if defect_class != "good"] # filter the good folder
        
        for class_idx, class_name in enumerate(defect_dirs):
            defect_class_names.append(class_name)
            defect_class_path = os.path.join(object_dir, class_name)
            for img_name in tqdm(os.listdir(defect_class_path)):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    img_path = os.path.join(defect_class_path, img_name)

                    try:
                        dino_feature = extract_extrafeatures(img_path, model, device)
                        
                        save_path = OUTPUT_DIR / object_type / class_name 
                        save_path.mkdir(parents=True, exist_ok=True)

                        np.save(save_path / str("feat_" + img_name.split(".")[0]), dino_feature)
                        np.save(save_path / str("label_" + img_name.split(".")[0]), class_idx)

                    except Exception as e:
                            print(f"Error at {img_name}: {e}")
        


def evaluate_object_cv(object_type: str, feature_root: str = "./dino_features",
                        n_splits: int = 5, clf: str = "svm"):
        """Runs full cross-validation for one object type and prints averaged F1."""

        # Detect number of folds (LOO or k-fold)
        object_dir     = Path(feature_root) / object_type
        n_originals    = sum(
            1 for class_dir in object_dir.iterdir() if class_dir.is_dir()
            for f in class_dir.glob("feat_*.npy") if "_aug" not in f.stem
        )
        actual_splits  = n_originals if n_originals < 20 else n_splits

        fold_f1s = []

        for fold in range(actual_splits):
            X_train, X_test, y_train_enc, y_test_enc = akf.load_features_for_object(
                feature_root, object_type,
                n_splits=n_splits, fold_index=fold
            )

            if clf == "svm":
                svm = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
                svm.fit(X_train, y_train_enc)
                y_pred = svm.predict(X_test)

                macro_f1 = f1_score(y_test_enc, y_pred, average="macro", zero_division=0)
                fold_f1s.append(macro_f1)
                print(f"  Fold {fold+1}/{actual_splits} — Macro F1: {macro_f1:.4f}")
            
            elif clf == "lr":
                # implement LogisticRegression Classifier
                lr = LogisticRegression(C=10, l1_ratio=0.3, class_weight="balanced", max_iter=500, solver='saga')
                lr.fit(X_train, y_train_enc)
                y_pred = lr.predict(X_test)

                # # try BayesSearch hyperparameter optimization
                # scaler = StandardScaler()
                # X_train_stdscaled = scaler.fit_transform(X_train)
                # X_test_stdscaled = scaler.transform(X_test)

                # params = {"C": Real(1e-3, 1e+2,prior='uniform'),
                #           "l1_ratio": Real(0.0, 1.0,prior='uniform')}
                # opt = BayesSearchCV(estimator=LogisticRegression(class_weight='balanced',solver='saga', max_iter=1000), search_spaces=params, scoring='f1_macro')
                # opt.fit(X_train_stdscaled, y_train_enc)
                # print(f"BayesSearch F1-Score: {opt.score(X_test_stdscaled,y_test_enc)}")
                # print(opt.best_params_)

                print(classification_report(y_test_enc, y_pred, zero_division=1))
                macro_f1 = f1_score(y_test_enc, y_pred, average="macro", zero_division=0)
                fold_f1s.append(macro_f1)
                print(f"  Fold {fold+1}/{actual_splits} — Macro F1: {macro_f1:.4f}")
            
            elif clf == "knn":
                # implement KNearestNeighbors Classifier
                knn = KNeighborsClassifier()
                knn.fit(X_train, y_train_enc)
                y_pred = knn.predict(X_test)

                macro_f1 = f1_score(y_test_enc, y_pred, average="macro", zero_division=0)
                fold_f1s.append(macro_f1)
                print(f"  Fold {fold+1}/{actual_splits} — Macro F1: {macro_f1:.4f}")
            
            elif clf == "rf":
                # implement RandomForest Classifier
                rf = RandomForestClassifier(class_weight="balanced")
                rf.fit(X_train, y_train_enc)
                y_pred = rf.predict(X_test)

                macro_f1 = f1_score(y_test_enc, y_pred, average="macro", zero_division=0)
                fold_f1s.append(macro_f1)
                print(f"  Fold {fold+1}/{actual_splits} — Macro F1: {macro_f1:.4f}")

        mean_f1 = np.mean(fold_f1s)
        std_f1  = np.std(fold_f1s)
        # print(f"\n  {object_type} — Final Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")
        return mean_f1, std_f1


def eval_object_bayessearchcv(object_type: str, feature_root: str = "./dino_features",
                        n_splits: int = 5, clf: str = "svm"):
    object_dir     = Path(feature_root) / object_type
    fold_f1s = []

    X_all, y_all, fold_idxs = akf.load_feature_indices_for_object(feature_root, object_type, n_splits=n_splits, random_state=42)

    # params = {"C": Real(1e-3, 1e+2,prior='uniform'), "l1_ratio": Real(0.0, 1.0,prior='uniform')}
    # estimator = LogisticRegression(class_weight='balanced',solver='saga', max_iter=1000)

    params = {'kernel': ['rbf', 'poly'],
              'C': Real(1e-3, 1e+3,prior='uniform'),
              'degree': [3,4,5,6,7]}
    estimator = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")

    opt = BayesSearchCV(estimator=estimator, search_spaces=params, scoring='f1_macro', cv=fold_idxs)
    opt.fit(X_all, y_all)
    print(f"Best {clf} F1 Score: {opt.best_score_}")
    print(opt.best_params_)
    

def average_classification_reports(reports: list[dict]) -> dict:
    """
    Average a list of sklearn classification_report dicts (output_dict=True).
    Handles nested keys (per-class dicts) and flat keys (accuracy).
    """
    # collect all values per (path) key
    sums = defaultdict(list)

    def collect(d, prefix=()):
        for k, v in d.items():
            if isinstance(v, dict):
                collect(v, prefix + (k,))
            else:
                sums[prefix + (k,)].append(v)

    for report in reports:
        collect(report)

    # rebuild nested dict with averaged (and std, optionally) values
    avg_report = {}
    for path, values in sums.items():
        d = avg_report
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n": len(values),  # how many folds contributed — flags missing classes
        }

    return avg_report 


def print_report_table_with_std(report: dict, digits: int = 2):
    summary_keys = {"accuracy", "macro avg", "weighted avg"}
    class_keys = [k for k in report.keys() if k not in summary_keys]

    headers = ["precision", "recall", "f1-score", "support"]
    name_width = max(len(k) for k in list(report.keys())) + 2
    col_width = 16

    # figure out the expected n (majority vote across all leaf entries)
    all_n = []
    def collect_n(d):
        for v in d.values():
            if isinstance(v, dict):
                if "n" in v:
                    all_n.append(v["n"])
                else:
                    collect_n(v)
    collect_n(report)
    expected_n = max(all_n) if all_n else None

    header_line = " " * name_width + "".join(f"{h:>{col_width}}" for h in headers) + "   n"
    print(header_line)
    print()

    def fmt(entry, is_support=False):
        if entry is None:
            return ""
        mean, std = entry["mean"], entry["std"]
        if is_support:
            return f"{mean:.1f} ± {std:.1f}"
        return f"{mean:.{digits}f} ± {std:.{digits}f}"

    def row_n(metrics):
        # all metrics in a row should share the same n (same folds contributed)
        # take n from the first available entry
        for h in ["precision", "recall", "f1-score", "support"]:
            if h in metrics:
                return metrics[h]["n"]
        return None

    def print_row(name, metrics):
        row = f"{name:<{name_width}}"
        for h in headers:
            val = fmt(metrics.get(h), is_support=(h == "support"))
            row += f"{val:>{col_width}}"

        n = row_n(metrics)
        flag = ""
        if expected_n is not None and n is not None and n < expected_n:
            flag = f"  ⚠ n={n}/{expected_n}"
        else:
            flag = f"  n={n}"
        print(row + flag)

    for k in class_keys:
        print_row(k, report[k])

    print()

    if "accuracy" in report:
        acc = report["accuracy"]
        support = report.get("macro avg", {}).get("support")
        row = f"{'accuracy':<{name_width}}"
        row += " " * col_width
        row += " " * col_width
        row += f"{fmt(acc):>{col_width}}"
        row += f"{fmt(support, is_support=True):>{col_width}}" if support else ""
        n = acc.get("n")
        flag = f"  n={n}" if (expected_n is None or n == expected_n) else f"  ⚠ n={n}/{expected_n}"
        print(row + flag)

    for k in ["macro avg", "weighted avg"]:
        if k in report:
            print_row(k, report[k])


def save_report_table(report: dict, filepath: str, digits: int = 2):
    """Save the formatted table as human-readable text."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_report_table_with_std(report, digits=digits)
    with open(filepath, "w") as f:
        f.write(buf.getvalue())

def save_report_json(report: dict, filepath: str):
    """Save the raw averaged report dict for later loading/comparison."""
    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)

def load_report_json(filepath: str) -> dict:
    with open(filepath) as f:
        return json.load(f)


def train_and_save_defect_classifier(object_type: str, feature_root: str = "./dino_features",
                        n_splits: int = 5, clf: str = "svm"):
    
    # get train and test split indices for the defect images, generated via the augkfold module
    X_all, y_all, fold_idxs = akf.load_feature_indices_for_object(feature_root, object_type, n_splits=n_splits, random_state=42)
    fold_f1s = []
    clf_report_dicts = []
    fold_counter = 1
    
    for fold in fold_idxs:
        X_train = X_all[fold[0]]
        X_test = X_all[fold[1]]
        y_train = y_all[fold[0]]
        y_test = y_all[fold[1]]

        svm = SVC(kernel="poly", degree=5, C=100, gamma="scale", class_weight="balanced")
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        
        clf_report = classification_report(y_test, y_pred, zero_division=1, output_dict=True)
        clf_report_dicts.append(clf_report)

        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        fold_f1s.append(macro_f1)
        # print(f" Fold {fold_counter} — Macro F1: {macro_f1:.4f}")
        fold_counter +=1

    avr_macro_f1 = np.sum(fold_f1s)/len(fold_f1s)
    print(f"Average macro F1 score: {avr_macro_f1}")
    avr_clf_report = average_classification_reports(clf_report_dicts)
    print_report_table_with_std(avr_clf_report)
    
    save_report_table(avr_clf_report, f"results/defect_clf_avg_report_{object_type}_{clf}.txt")
    save_report_json(avr_clf_report, f"results/defect_clf_avg_report_{object_type}_{clf}.json")

    ### create deployable SVM and save it
    dep_svm = SVC(kernel="poly", degree=5, C=100, gamma="scale", class_weight="balanced")
    dep_svm.fit(X_all, y_all)

    os.makedirs("models/defect_models", exist_ok=True)
    joblib.dump(dep_svm, f"models/defect_models/dinov2_svm_{object_type}_final.joblib")

    # NOTE: load later, would go like this
    # final_svm = joblib.load("defect_models/dinov2_svm_{object_type}_final.joblib")
    # preds = final_svm.predict(X_new)




if __name__ == "__main__":
    
    ### STEP 1: get the DINOv2 combined feature vectors for all object categories
    # for categ in ALL_OBJECT_TYPES:
    #     generate_DINOv2_features(categ)
    
    ### STEP 2: train the defect classifier and save a model for later use
    for categ in ALL_OBJECT_TYPES:
        if categ != "toothbrush":
            train_and_save_defect_classifier(categ)
