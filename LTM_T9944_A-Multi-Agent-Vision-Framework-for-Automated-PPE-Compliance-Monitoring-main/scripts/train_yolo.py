"""
Train YOLO PPE Detection Model
Project:
A-Multi-Agent-Vision-Framework-for-Automated-PPE-Compliance-Monitoring

Dataset:
Ultralytics Construction-PPE

Environment:
- Python 3.11
- Ultralytics 8.4.155
- PyTorch CPU

What this script does:
1. Downloads Construction-PPE dataset if it is not already present.
2. Creates a local dataset YAML.
3. Loads a pretrained YOLO nano model.
4. Trains the PPE detector on CPU.
5. Validates the trained model.
6. Evaluates the test split.
7. Copies best.pt to:
      models/ppe_yolo.pt
8. Verifies that the final model can be loaded.
"""

from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile

from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
PPE_DIR = DATA_DIR / "construction_ppe"

MODELS_DIR = ROOT_DIR / "models"
RUNS_DIR = ROOT_DIR / "runs" / "detect"

DATASET_YAML = PPE_DIR / "construction_ppe.yaml"

FINAL_MODEL = MODELS_DIR / "ppe_yolo.pt"

# Official Ultralytics Construction-PPE archive
DATASET_URL = (
    "https://github.com/ultralytics/assets/releases/"
    "download/v0.0.0/construction-ppe.zip"
)

DATASET_ZIP = DATA_DIR / "construction-ppe.zip"

# Small pretrained model.
# This is intentionally nano-sized because your machine is CPU-only.
BASE_MODEL = "yolo11n.pt"

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

EPOCHS = 50
IMAGE_SIZE = 640

# CPU-friendly batch size.
# 4 is safer for a normal laptop.
BATCH_SIZE = 4

# CPU worker count.
# Windows + CPU is more stable with 0.
WORKERS = 0

# Reproducibility
SEED = 42


# ============================================================
# LOGGING HELPERS
# ============================================================

def print_header(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_step(step: int, title: str):
    print()
    print(f"[STEP {step}] {title}")
    print("-" * 70)


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories():
    """Create required project directories."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PPE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET DOWNLOAD
# ============================================================

def download_dataset():
    """
    Download the official Construction-PPE dataset.

    The dataset is approximately 178 MB.
    """

    print_step(1, "Checking Construction-PPE dataset")

    expected_train_dir = PPE_DIR / "images" / "train"
    expected_val_dir = PPE_DIR / "images" / "val"
    expected_test_dir = PPE_DIR / "images" / "test"

    if (
        expected_train_dir.exists()
        and expected_val_dir.exists()
        and expected_test_dir.exists()
    ):
        print("Construction-PPE dataset already exists.")
        print(f"Dataset location: {PPE_DIR}")
        return

    print("Construction-PPE dataset not found.")
    print("Downloading official dataset...")
    print()
    print(f"URL: {DATASET_URL}")
    print(f"Destination: {DATASET_ZIP}")

    try:
        urllib.request.urlretrieve(
            DATASET_URL,
            DATASET_ZIP,
        )

    except Exception as exc:
        print()
        print("ERROR: Dataset download failed.")
        print(f"Reason: {exc}")
        sys.exit(1)

    print()
    print("Dataset download completed.")

    # --------------------------------------------------------
    # Extract ZIP
    # --------------------------------------------------------

    print("Extracting dataset...")

    try:
        with zipfile.ZipFile(DATASET_ZIP, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)

    except Exception as exc:
        print()
        print("ERROR: Dataset extraction failed.")
        print(f"Reason: {exc}")
        sys.exit(1)

    print("Dataset extraction completed.")

    # --------------------------------------------------------
    # Cleanup ZIP
    # --------------------------------------------------------

    if DATASET_ZIP.exists():
        try:
            DATASET_ZIP.unlink()
            print("Temporary ZIP file removed.")
        except Exception:
            print("Warning: Could not remove temporary ZIP file.")

    # --------------------------------------------------------
    # Validate extracted structure
    # --------------------------------------------------------

    if not expected_train_dir.exists():
        print()
        print("ERROR: Training directory was not found after extraction.")
        print(f"Expected: {expected_train_dir}")
        sys.exit(1)

    if not expected_val_dir.exists():
        print()
        print("ERROR: Validation directory was not found after extraction.")
        print(f"Expected: {expected_val_dir}")
        sys.exit(1)

    if not expected_test_dir.exists():
        print()
        print("ERROR: Test directory was not found after extraction.")
        print(f"Expected: {expected_test_dir}")
        sys.exit(1)

    print()
    print("Dataset structure verified.")


# ============================================================
# DATASET YAML
# ============================================================

def create_dataset_yaml():
    """
    Create a project-local Construction-PPE YAML.

    Keeping a local YAML makes the project reproducible and
    avoids depending on an installed Ultralytics package path.
    """

    print_step(2, "Creating local dataset YAML")

    yaml_content = """# Construction-PPE Dataset
# Project:
# A-Multi-Agent-Vision-Framework-for-Automated-PPE-Compliance-Monitoring

path: construction_ppe

train: images/train
val: images/val
test: images/test

names:
  0: helmet
  1: gloves
  2: vest
  3: boots
  4: goggles
  5: none
  6: Person
  7: no_helmet
  8: no_goggle
  9: no_gloves
  10: no_boots
"""

    DATASET_YAML.write_text(
        yaml_content,
        encoding="utf-8",
    )

    print(f"Dataset YAML created:")
    print(DATASET_YAML)


# ============================================================
# DATASET STATISTICS
# ============================================================

def count_images():
    """Print image counts for train/val/test."""

    print_step(3, "Checking dataset images")

    splits = {
        "train": PPE_DIR / "images" / "train",
        "val": PPE_DIR / "images" / "val",
        "test": PPE_DIR / "images" / "test",
    }

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    total = 0

    for split_name, split_dir in splits.items():

        if not split_dir.exists():
            print(f"{split_name}: DIRECTORY NOT FOUND")
            continue

        count = sum(
            1
            for file in split_dir.rglob("*")
            if file.is_file()
            and file.suffix.lower() in extensions
        )

        print(f"{split_name:>5}: {count} images")
        total += count

    print("-" * 30)
    print(f"Total : {total} images")

    if total == 0:
        print()
        print("ERROR: No images found.")
        sys.exit(1)


# ============================================================
# MODEL DOWNLOAD
# ============================================================

def load_base_model():
    """
    Load pretrained YOLO nano model.

    Ultralytics downloads the pretrained model automatically
    the first time it is requested.
    """

    print_step(4, "Loading pretrained YOLO model")

    print(f"Base model: {BASE_MODEL}")
    print("If the model is not present, Ultralytics will download it.")

    try:
        model = YOLO(BASE_MODEL)

    except Exception as exc:
        print()
        print("ERROR: Could not load pretrained YOLO model.")
        print(f"Reason: {exc}")
        sys.exit(1)

    print("Pretrained YOLO model loaded successfully.")

    return model


# ============================================================
# TRAINING
# ============================================================

def train_model(model):
    """Train YOLO PPE detector."""

    print_step(5, "Training PPE detection model")

    print("Training configuration:")
    print(f"  Dataset      : {DATASET_YAML}")
    print(f"  Model        : {BASE_MODEL}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  Image size   : {IMAGE_SIZE}")
    print(f"  Batch size   : {BATCH_SIZE}")
    print(f"  Workers      : {WORKERS}")
    print(f"  Device       : CPU")
    print(f"  Seed         : {SEED}")

    print()
    print("CPU training may take a while.")
    print("Do not close the terminal while training is running.")
    print()

    try:
        results = model.train(
            data=str(DATASET_YAML),
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device="cpu",
            workers=WORKERS,
            seed=SEED,

            # Output
            project=str(RUNS_DIR),
            name="ppe_training",
            exist_ok=True,

            # Training behavior
            pretrained=True,
            patience=15,

            # Save checkpoints
            save=True,
            save_period=10,

            # Validation
            val=True,

            # Cache disabled to reduce RAM usage
            cache=False,

            # Verbose training logs
            verbose=True,
        )

    except Exception as exc:
        print()
        print("ERROR: YOLO training failed.")
        print(f"Reason: {exc}")
        sys.exit(1)

    print()
    print("Training completed successfully.")

    return results


# ============================================================
# LOCATE BEST MODEL
# ============================================================

def locate_best_model():
    """Locate best.pt generated by training."""

    print_step(6, "Locating trained model")

    best_model = (
        RUNS_DIR
        / "ppe_training"
        / "weights"
        / "best.pt"
    )

    last_model = (
        RUNS_DIR
        / "ppe_training"
        / "weights"
        / "last.pt"
    )

    if best_model.exists():
        print(f"best.pt found:")
        print(best_model)
        return best_model

    if last_model.exists():
        print()
        print("WARNING: best.pt was not found.")
        print("last.pt exists, but this script will NOT rename it as best.pt.")
        print(f"last.pt: {last_model}")
        sys.exit(1)

    print()
    print("ERROR: Neither best.pt nor last.pt was found.")
    print(f"Expected directory: {RUNS_DIR / 'ppe_training' / 'weights'}")
    sys.exit(1)


# ============================================================
# VALIDATE MODEL
# ============================================================

def validate_model(best_model):
    """Run validation on the validation split."""

    print_step(7, "Validating trained model")

    try:
        trained_model = YOLO(str(best_model))

        metrics = trained_model.val(
            data=str(DATASET_YAML),
            split="val",
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device="cpu",
            workers=WORKERS,
            plots=True,
        )

    except Exception as exc:
        print()
        print("ERROR: Validation failed.")
        print(f"Reason: {exc}")
        sys.exit(1)

    print()
    print("Validation completed.")

    try:
        print(f"mAP50      : {metrics.box.map50:.4f}")
        print(f"mAP50-95   : {metrics.box.map:.4f}")
    except Exception:
        print("Validation metrics were generated.")
        print("See the generated results directory for details.")


# ============================================================
# TEST SET EVALUATION
# ============================================================

def evaluate_test_set(best_model):
    """Evaluate final model on the held-out test split."""

    print_step(8, "Evaluating test split")

    try:
        trained_model = YOLO(str(best_model))

        metrics = trained_model.val(
            data=str(DATASET_YAML),
            split="test",
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device="cpu",
            workers=WORKERS,
            plots=True,
        )

    except Exception as exc:
        print()
        print("ERROR: Test evaluation failed.")
        print(f"Reason: {exc}")
        sys.exit(1)

    print()
    print("Test evaluation completed.")

    try:
        print(f"Test mAP50    : {metrics.box.map50:.4f}")
        print(f"Test mAP50-95 : {metrics.box.map:.4f}")
    except Exception:
        print("Test metrics were generated.")


# ============================================================
# COPY FINAL MODEL
# ============================================================

def install_final_model(best_model):
    """
    Copy best.pt to:

        models/ppe_yolo.pt
    """

    print_step(9, "Installing final PPE model")

    try:
        shutil.copy2(
            best_model,
            FINAL_MODEL,
        )

    except Exception as exc:
        print()
        print("ERROR: Could not copy trained model.")
        print(f"Reason: {exc}")
        sys.exit(1)

    if not FINAL_MODEL.exists():
        print("ERROR: Final model was not created.")
        sys.exit(1)

    size_mb = FINAL_MODEL.stat().st_size / (1024 * 1024)

    print("Final model created successfully:")
    print(FINAL_MODEL)
    print(f"Model size: {size_mb:.2f} MB")


# ============================================================
# FINAL MODEL VERIFICATION
# ============================================================

def verify_final_model():
    """Load final model and print class names."""

    print_step(10, "Verifying final PPE model")

    try:
        model = YOLO(str(FINAL_MODEL))

        print()
        print("Model loaded successfully.")
        print()
        print("Detected classes:")

        for class_id, class_name in model.names.items():
            print(f"  {class_id:>2} -> {class_name}")

    except Exception as exc:
        print()
        print("ERROR: Final model verification failed.")
        print(f"Reason: {exc}")
        sys.exit(1)


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "PPE YOLO TRAINING PIPELINE\n"
        "A-Multi-Agent-Vision-Framework-for-Automated-PPE-Compliance-Monitoring"
    )

    print(f"Project root : {ROOT_DIR}")
    print(f"Python       : {sys.version.split()[0]}")

    create_directories()

    # 1. Download dataset
    download_dataset()

    # 2. Create YAML
    create_dataset_yaml()

    # 3. Check images
    count_images()

    # 4. Load pretrained model
    model = load_base_model()

    # 5. Train
    train_model(model)

    # 6. Locate best.pt
    best_model = locate_best_model()

    # 7. Validate
    validate_model(best_model)

    # 8. Test
    evaluate_test_set(best_model)

    # 9. Copy to models/
    install_final_model(best_model)

    # 10. Verify
    verify_final_model()

    print_header("PPE MODEL PIPELINE COMPLETED")

    print()
    print("Final trained model:")
    print(FINAL_MODEL)

    print()
    print("Training results:")
    print(RUNS_DIR / "ppe_training")

    print()
    print("The model is now ready for the Vision Agent.")


if __name__ == "__main__":
    main()