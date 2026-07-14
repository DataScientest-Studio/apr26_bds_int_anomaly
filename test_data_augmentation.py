from torchvision import transforms
import torchvision.transforms.functional as TF
import random
from PIL import Image
from pathlib import Path

root_path = r"C:\Users\Tabea\Desktop\Liora_Weiterbildung\Projects\Group_Project\MVTec_AD\archive_aug"
root_path = root_path.replace("\\", "/")

DATA_ROOT     = Path(root_path)

ALL_OBJECT_TYPES = [
    "bottle","cable","capsule","carpet","grid","hazelnut",
    "leather","metal_nut","pill","screw","tile","toothbrush",
    "transistor","wood","zipper"
]



def augment_pair(image: Image.Image, mask: Image.Image):
    """
    Wendet dieselbe zufällige Transformation auf Bild UND Maske an.
    Wichtig: nur geometrische Transforms für die Maske, kein ColorJitter.
    """
    # Zufällige Parameter einmalig bestimmen
    angle        = random.uniform(-10, 10)
    flip_h       = random.random() > 0.5
    flip_v       = random.random() > 0.5
    translate_x  = random.uniform(-0.05, 0.05)
    translate_y  = random.uniform(-0.05, 0.05)
    scale        = random.uniform(0.95, 1.05)

    # Geometrische Transforms auf beide anwenden
    image = TF.affine(image, angle=angle,
                      translate=[int(translate_x * image.width),
                                 int(translate_y * image.height)],
                      scale=scale, shear=0)
    mask  = TF.affine(mask,  angle=angle,
                      translate=[int(translate_x * mask.width),
                                 int(translate_y * mask.height)],
                      scale=scale, shear=0)

    if flip_h:
        image = TF.hflip(image)
        mask  = TF.hflip(mask)

    if flip_v:
        image = TF.vflip(image)
        mask  = TF.vflip(mask)

    # ColorJitter NUR auf das Bild — Maske bleibt binär
    brightness = random.uniform(0.9, 1.1)
    contrast   = random.uniform(0.9, 1.1)
    image = TF.adjust_brightness(image, brightness)
    image = TF.adjust_contrast(image, contrast)

    return image, mask



def augment_dataset_offline(object_type: str, n_copies: int = 5):
    test_dir         = DATA_ROOT / object_type / "test"
    ground_truth_dir = DATA_ROOT / object_type / "ground_truth"

    for class_dir in test_dir.iterdir():
        if not class_dir.is_dir() or class_dir.name == "good":
            continue

        mask_class_dir = ground_truth_dir / class_dir.name
        original_images = [p for p in class_dir.glob("*.png") if "_aug" not in p.stem]

        print(f"  {object_type}/{class_dir.name}: "
              f"{len(original_images)} Originale → "
              f"{len(original_images) * (1 + n_copies)} nach Augmentierung")

        for img_path in original_images:
            mask_path = mask_class_dir / f"{img_path.stem}_mask.png"

            img  = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L") if mask_path.exists() else None

            for i in range(n_copies):
                if mask is not None:
                    aug_img, aug_mask = augment_pair(img, mask)
                else:
                    # kein Maskenpfad → nur Bild augmentieren
                    aug_img, _ = augment_pair(img, Image.new("L", img.size, 0))
                    aug_mask   = None

                # Augmentiertes Bild speichern
                aug_img.save(class_dir / f"{img_path.stem}_aug{i}.png")

                # Augmentierte Maske speichern — gleicher Name + _mask
                if aug_mask is not None:
                    aug_mask.save(mask_class_dir / f"{img_path.stem}_aug{i}_mask.png")

if __name__ == "__main__":
   
    for object_type in ALL_OBJECT_TYPES:
        augment_dataset_offline(object_type, n_copies= 10)