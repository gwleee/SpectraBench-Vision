#!/usr/bin/env python3
"""
Patch LLaVA files to fix missing imports.

1. builder.py - Add missing direct imports for model classes
2. llava/__init__.py - Remove invalid imports that cause ImportError
"""
import pathlib
import sys

def patch_builder():
    """Patch builder.py to add missing imports."""
    builder_path = pathlib.Path('/workspace/LLaVA/llava/model/builder.py')

    if not builder_path.exists():
        print(f"ERROR: {builder_path} not found")
        sys.exit(1)

    content = builder_path.read_text()

    # Check if already patched
    if 'from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM' in content:
        print("✓ builder.py already patched")
        return

    lines = content.split('\n')

    # Find the transformers import line
    import_idx = None
    for i, line in enumerate(lines):
        if 'from transformers import' in line:
            import_idx = i
            break

    if import_idx is None:
        print("ERROR: Could not find 'from transformers import' line")
        sys.exit(1)

    # Insert the three imports after the transformers import
    imports_to_add = [
        'from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM',
        'from llava.model.language_model.llava_mpt import LlavaMptForCausalLM',
        'from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM'
    ]

    # Insert in reverse order to maintain correct line positions
    for import_line in reversed(imports_to_add):
        lines.insert(import_idx + 1, import_line)

    # Write back
    builder_path.write_text('\n'.join(lines))

    print("✓ Successfully patched builder.py")
    print(f"  Added {len(imports_to_add)} import statements")

def patch_llava_init():
    """Remove problematic imports from llava/__init__.py."""
    init_path = pathlib.Path('/workspace/LLaVA/llava/__init__.py')

    if not init_path.exists():
        print(f"ERROR: {init_path} not found")
        sys.exit(1)

    content = init_path.read_text()

    # Check if already patched (file should be empty or have no .model imports)
    if 'from .model import' not in content:
        print("✓ llava/__init__.py already patched (no .model imports)")
        return

    # Remove all lines that import from .model
    lines = content.split('\n')
    cleaned_lines = [line for line in lines if 'from .model import' not in line]

    # Write back
    init_path.write_text('\n'.join(cleaned_lines))

    print("✓ Successfully patched llava/__init__.py")
    print(f"  Removed {len(lines) - len(cleaned_lines)} import lines")

if __name__ == '__main__':
    patch_builder()
    patch_llava_init()
    print("\n✓ All patches applied successfully")
