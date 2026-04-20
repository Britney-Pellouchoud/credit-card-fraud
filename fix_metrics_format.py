import os
import re

ROOT_DIR = "training_alg"

TARGET_PATTERN = r'print\("METRICS_START".*results.*"METRICS_END"\)'


def fix_file(path):
    with open(path, "r") as f:
        content = f.read()

    if "METRICS_START" not in content:
        return False

    # replace bad pattern with correct one
    new_content = re.sub(
        TARGET_PATTERN,
        'print("METRICS_START", json.dumps(results), "METRICS_END")',
        content,
    )

    # also ensure json import exists
    if "json.dumps" in new_content and "import json" not in new_content:
        new_content = "import json\n" + new_content

    if new_content != content:
        with open(path, "w") as f:
            f.write(new_content)
        return True

    return False


def walk_and_fix():
    changed_files = []

    for root, _, files in os.walk(ROOT_DIR):
        for file in files:
            if file.startswith("train_") and file.endswith(".py"):
                path = os.path.join(root, file)
                if fix_file(path):
                    changed_files.append(path)

    print("\n✅ Fixed files:")
    for f in changed_files:
        print(" -", f)

    print(f"\nTotal: {len(changed_files)} files updated")


if __name__ == "__main__":
    walk_and_fix()
