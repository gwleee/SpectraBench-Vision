#!/usr/bin/env python3
"""
Runtime patch for VLMEvalKit to work with transformers 4.33
Disables Thyme model which requires transformers 4.35+
"""
import os
import sys

def patch_vlmeval_433():
    """Patch VLMEvalKit to skip Thyme model import and config"""
    success = True

    # Patch 1: vlm/__init__.py
    vlm_init_path = "/workspace/VLMEvalKit/vlmeval/vlm/__init__.py"
    if os.path.exists(vlm_init_path):
        with open(vlm_init_path, 'r') as f:
            content = f.read()

        if "# PATCHED FOR TRANSFORMERS 4.33" not in content:
            original_import = "from .thyme import Thyme"
            patched_import = """# PATCHED FOR TRANSFORMERS 4.33 COMPATIBILITY
try:
    from .thyme import Thyme
except (ImportError, ModuleNotFoundError) as e:
    print(f"⚠️  Skipping Thyme model (requires transformers 4.35+): {e}")
    Thyme = None"""

            if original_import in content:
                content = content.replace(original_import, patched_import)
                with open(vlm_init_path, 'w') as f:
                    f.write(content)
                print("✅ Patched vlm/__init__.py")
            else:
                print("⚠️  vlm/__init__.py already modified or Thyme import not found")
        else:
            print("✅ vlm/__init__.py already patched")
    else:
        print(f"❌ vlm/__init__.py not found")
        success = False

    # Patch 2: config.py - comment out Thyme model entries
    config_path = "/workspace/VLMEvalKit/vlmeval/config.py"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            lines = f.readlines()

        if not any("# PATCHED FOR TRANSFORMERS 4.33" in line for line in lines):
            new_lines = []
            for line in lines:
                # Comment out Thyme model definitions
                if "Thyme" in line and ("partial(Thyme" in line or '"Thyme' in line):
                    new_lines.append(f"    # PATCHED FOR TRANSFORMERS 4.33: {line.lstrip()}")
                else:
                    new_lines.append(line)

            with open(config_path, 'w') as f:
                f.writelines(new_lines)
            print("✅ Patched config.py")
        else:
            print("✅ config.py already patched")
    else:
        print(f"❌ config.py not found")
        success = False

    return success

if __name__ == "__main__":
    success = patch_vlmeval_433()
    sys.exit(0 if success else 1)
