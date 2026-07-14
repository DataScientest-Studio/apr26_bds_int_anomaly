import warnings
# import os
# os.environ["PYTHONUTF8"] = "1"
# surpress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

import torch

# activate hardware acceleration for matrix-multiplication
torch.set_float32_matmul_precision('high')

from anomalib.data import MVTecAD, Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.deploy import ExportType  # neu importiert für das speichern
from anomalib.metrics import F1Score, AUROC, AUPRO
from anomalib.metrics import Evaluator
from anomalib.data.utils import ValSplitMode

from lightning.pytorch.loggers import CSVLogger
from pathlib import Path


DATASET_ROOT_PATH = r"C:\Users\Tabea\Desktop\Liora_Weiterbildung\Projects\Group_Project\MVTec_AD\archive"


# list of all the onbject types in the MVTec AD Dataset
categories = ['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']

# get all the defect folders for one category
def get_defect_folder(dataset_path, categorie:str):
    test_dir = Path(dataset_path) / categorie / "test"
    defect_paths = [
        str(p.relative_to(dataset_path)) 
        for p in test_dir.iterdir() 
        if p.is_dir() and p.name != "good"
    ]  
    return defect_paths

# train the PatchCore model for one categorie and save the model via openvino
def train_patchcore(categorie:str):

    # Initialize metrics with specific fields
    image_f1_score = F1Score(fields=["pred_label", "gt_label"], prefix="image_")
    pixel_f1_score = F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_")

    aupro_score = AUPRO(fields=["anomaly_map","gt_mask"], prefix="aupro_")
    
    image_auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
    pixel_auroc = AUROC(fields=["anomaly_map","gt_mask"], prefix="pixel_")


    # Create evaluator with test metrics (for validation, use val_metrics arg)
    evaluator = Evaluator(test_metrics=[image_f1_score, pixel_f1_score, image_auroc, pixel_auroc, aupro_score]) #, image_f1_thresh, pixel_f1_thresh])

    # 1. hardware-check (amdgpu via rocm windows)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # 2. load data module
    # <-- that is a user specific path for the place where the dataset is saved (on a Windows machine)
    root_path = DATASET_ROOT_PATH
    root_path = root_path.replace("\\", "/")

    # define data logger
    csv_logger = CSVLogger(save_dir=f"./results/patchcore_{categorie}", name="anomalie_test")

    # load the specific dataset via anomalib Folder module
    datamodule = Folder(
        name=f"{categorie}",
        root=root_path,                                         # main folder for the dataset
        normal_dir=f"{categorie}/train/good",                   # folder for all the good training sample images
        abnormal_dir=get_defect_folder(root_path, categorie),   # get all the paths to the defect objects
        normal_test_dir=f"{categorie}/test/good",               # folder for the good test validation images
        mask_dir=f"{categorie}/ground_truth",                   # folder for the groundtruth masks of defects
        val_split_mode= ValSplitMode.SAME_AS_TEST,              # use same images for validation as for the test
        val_split_ratio = 1.0                                   # use 100% of test images for validation
    )

    # 3. initialize the PatchCore model
    model = Patchcore(evaluator=evaluator)

    # 4. configure the anomalib engine
    engine = Engine(
        accelerator=accelerator,
        devices=1,
        logger=csv_logger
    )

    # 5. train the model
    engine.fit(model=model, datamodule=datamodule)

    # 6. save the model into a "./results/" subfolder
    # dies erstellt einen ordner "results" in ihrem projektverzeichnis
    engine.export(
        model=model,
        export_type=ExportType.OPENVINO,  # openvino ist extrem schnell bei der bildauswertung
        export_root=f"./results/patchcore_{categorie}"
    )

    print(f"\n\nThe model was saved succesfully under ./results/patchcore_{categorie} !\n\n")

    # test the model
    engine.test(model, datamodule=datamodule)

if __name__ == '__main__':

    # train multiple PatchCore models for each object categorie
    for cat in categories:
        train_patchcore(cat)

    # train the PatchCore model for one categorie
    # train_patchcore('bottle')