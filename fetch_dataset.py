from pathlib import Path
import shutil
import sys
import kagglehub

# https://www.kaggle.com/datasets/disisbig/hindi-wikipedia-articles-55k
DATASET_ID = "disisbig/hindi-wikipedia-articles-55k"
TARGET_DIR = Path("./data")

MARKER_FILE = TARGET_DIR / ".FETCHED_FROM_KAGGLEHUB"

COLOR_CYAN = "\033[1;3;36m"
COLOR_RESET = "\033[0m"


def fetch_detaset() -> None:
    # Guard: dataset already fetched
    if TARGET_DIR.exists() and MARKER_FILE.exists():
        print(
            f"[OK] Dataset already present at: {COLOR_CYAN}{TARGET_DIR.resolve()}{COLOR_RESET}"
        )
        return

    if TARGET_DIR.exists() and not MARKER_FILE.exists():
        print(
            "[ERROR] Target directory exists but marker file is missing.\n"
            "This likely means the directory was created manually.\n"
            "Refusing to overwrite."
        )
        sys.exit(1)

    print("[INFO] Downloading dataset via kagglehub...")
    src = Path(kagglehub.dataset_download(DATASET_ID)).resolve()

    print(f"[INFO] Moving dataset from cache:\n  {src}")
    print(
        f"[INFO] To project directory:\n  {COLOR_CYAN}{TARGET_DIR.resolve()}{COLOR_RESET}"
    )

    shutil.move(src, TARGET_DIR)

    # create marker to prevent future kagglehub calls
    MARKER_FILE.write_text(
        f"dataset={DATASET_ID}\nnote=Moved from kagglehub cache. Do not re-download!\n"
    )

    print("[DONE] Dataset fetched and locked.")


if __name__ == "__main__":
    fetch_detaset()
