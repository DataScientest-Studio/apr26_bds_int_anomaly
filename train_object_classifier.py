import torch
import torchvision.transforms as T
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import joblib

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re

DATASET_ROOT_PATH = r"C:\Users\Tabea\Desktop\Liora_Weiterbildung\Projects\Group_Project\MVTec_AD\archive_aug"

# 1) get all the defect images of one object type and extract the feature vectors
root_path = DATASET_ROOT_PATH
root_path = root_path.replace("\\", "/")
root_path = Path(root_path)
DATA_DIR = root_path
# OUTPUT_DIR = Path("./good_dino_features")

import json


GOOD_FEATURES_DIR = Path("./good_dino_features")

def save_model(model, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)

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



def generate_DINOv2_router_features(object_types=None, model=None, device=None):
    """Extrahiert DINOv2 Features aus train/good Bildern aller Objekttypen
    für den Object-Type-Router-Klassifikator. Speichert getrennt von den
    Defekt-Features unter ./good_dino_features.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device).eval()

    if object_types is None:
        object_types = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])

    GOOD_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    for class_idx, object_type in enumerate(object_types):
        good_dir = DATA_DIR / object_type / "train" / "good"

        if not good_dir.exists():
            print(f"Skipping {object_type}: {good_dir} not found")
            continue

        save_path = GOOD_FEATURES_DIR / object_type
        save_path.mkdir(parents=True, exist_ok=True)

        for img_name in tqdm(os.listdir(good_dir), desc=object_type):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                if "_aug" in img_name:
                    continue

                img_path = good_dir / img_name

                try:
                    dino_feature = extract_extrafeatures(img_path, model, device)

                    np.save(save_path / f"feat_{img_name.split('.')[0]}", dino_feature)
                    np.save(save_path / f"label_{img_name.split('.')[0]}", class_idx)

                except Exception as e:
                    print(f"Error at {img_name}: {e}")

    label_map = {idx: name for idx, name in enumerate(object_types)}
    with open(GOOD_FEATURES_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"Saved router features to {GOOD_FEATURES_DIR}")
    return label_map


def load_router_features(features_dir=GOOD_FEATURES_DIR):
    """Lädt alle gespeicherten Feature-/Label-Paare aus ./good_dino_features
    und gibt X, y sowie das label_map (class_idx -> object_type) zurück.
    """
    features_dir = Path(features_dir)

    with open(features_dir / "label_map.json") as f:
        label_map = json.load(f)
        label_map = {int(k): v for k, v in label_map.items()}

    X, y = [], []

    for object_dir in sorted(features_dir.iterdir()):
        if not object_dir.is_dir():
            continue

        feat_files = sorted(object_dir.glob("feat_*.npy"))

        for feat_path in feat_files:
            stem = feat_path.name.replace("feat_", "", 1).replace(".npy", "")
            label_path = object_dir / f"label_{stem}.npy"

            if not label_path.exists():
                print(f"Missing label for {feat_path}, skipping")
                continue

            X.append(np.load(feat_path))
            y.append(int(np.load(label_path)))

    X = np.stack(X)
    y = np.array(y)

    return X, y, label_map



def load_defect_features_for_router_test(features_dir="./dino_features", label_map=None):
    """Lädt alle DINOv2-Features aus ./dino_features/{object}/{defect}/,
    gelabelt mit dem Objekttyp (nicht dem Defekttyp) — zum Testen des
    Object-Type-Routers auf Defektbildern. Augmentierte Bilder (_aug*) werden ausgeschlossen.
    """
    features_dir = Path(features_dir)

    if label_map is None:
        with open(GOOD_FEATURES_DIR / "label_map.json") as f:
            label_map = json.load(f)
            label_map = {int(k): v for k, v in label_map.items()}

    name_to_idx = {name: idx for idx, name in label_map.items()}

    # matches feat_000.npy, feat_123.npy — NOT feat_000_aug0.npy
    feat_pattern = re.compile(r"^feat_\d+\.npy$")

    X, y_true, meta = [], [], []

    for object_dir in sorted(features_dir.iterdir()):
        if not object_dir.is_dir():
            continue

        object_type = object_dir.name
        if object_type not in name_to_idx:
            print(f"Skipping {object_type}: not in router label_map")
            continue

        for defect_dir in sorted(object_dir.iterdir()):
            if not defect_dir.is_dir():
                continue

            for feat_path in sorted(defect_dir.iterdir()):
                if not feat_pattern.match(feat_path.name):
                    continue

                X.append(np.load(feat_path))
                y_true.append(name_to_idx[object_type])
                meta.append((object_type, defect_dir.name, feat_path.name))

    X = np.stack(X)
    y_true = np.array(y_true)

    return X, y_true, meta


if __name__ == "__main__":

    # STEP 1: extraction (run once)
    # label_map = generate_DINOv2_router_features()

    # STEP 2: train and save the object classifier
    # training later, possibly in a different script/session
    # X, y, label_map = load_router_features()
    # print(label_map)

    # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    # router_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    # router_clf.fit(X_train, y_train)

    # y_pred = router_clf.predict(X_test)
    # print(classification_report(y_test, y_pred))

    # os.makedirs("models/object_model", exist_ok=True)
    # save_model(router_clf, "models/object_model/object_type_router.joblib")

    # Step 3: test against the defect images
    X, y_true, meta = load_defect_features_for_router_test()
    router_clf = joblib.load("models/object_model/object_type_router.joblib")
    y_pred = router_clf.predict(X)
    print(classification_report(y_true, y_pred))

    ### Object router is expected to predict the object type correctly 100% of the time for good aswell as for defect images