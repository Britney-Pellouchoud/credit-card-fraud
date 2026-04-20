import os
import re

ROOT_DIR = "training_alg"


def extract_main_block(content):
    """
    Very simple heuristic: grab everything under if __name__ == "__main__"
    """
    pattern = r'if __name__ == "__main__":(.*)'
    match = re.search(pattern, content, re.DOTALL)

    if not match:
        return None

    return match.group(1)


def wrap_as_function(main_block):
    """
    Wraps main block into run_experiment()
    and fixes indentation.
    """

    lines = main_block.split("\n")
    indented = []

    for line in lines:
        if line.strip():
            indented.append("    " + line)
        else:
            indented.append("")

    function = (
        "\n\ndef run_experiment():\n" + "\n".join(indented) + "\n    return results\n"
    )

    return function


def fix_file(path):
    with open(path, "r") as f:
        content = f.read()

    if "__main__" not in content:
        return False

    main_block = extract_main_block(content)
    if not main_block:
        return False

    function_block = wrap_as_function(main_block)

    # remove old main block
    content = re.sub(r'if __name__ == "__main__":.*', "", content, flags=re.DOTALL)

    # add run_experiment above final section
    new_content = content.strip() + "\n" + function_block + "\n"

    # add safe entrypoint
    new_content += '\n\nif __name__ == "__main__":\n' "    print(run_experiment())\n"

    with open(path, "w") as f:
        f.write(new_content)

    return True


def run():
    changed = []

    for root, _, files in os.walk(ROOT_DIR):
        for file in files:
            if file.startswith("train_") and file.endswith(".py"):
                path = os.path.join(root, file)

                if fix_file(path):
                    changed.append(path)

    print("\n✅ Converted files:")
    for f in changed:
        print(" -", f)

    print(f"\nTotal: {len(changed)} files updated")


if __name__ == "__main__":
    run()
