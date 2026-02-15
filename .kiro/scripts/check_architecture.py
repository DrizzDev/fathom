#!/usr/bin/env python3
"""
Architectural boundary enforcement script.

This script checks that imports respect the hexagonal architecture boundaries:
1. Core layer cannot import from adapters
2. Interfaces layer can only import from schemas
3. Strategies should receive ports via dependency injection (no direct adapter imports)
4. Adapters can import from anywhere (they're at the edge)
5. Processing can only import from schemas and utils
"""

import ast
import sys
from pathlib import Path
from typing import List, Set, Tuple


class ImportChecker(ast.NodeVisitor):
    """AST visitor to check imports."""

    def __init__(self, file_path: Path, src_root: Path):
        self.file_path = file_path
        self.src_root = src_root
        self.violations: List[Tuple[int, str]] = []
        
        # Determine which layer this file belongs to
        rel_path = file_path.relative_to(src_root)
        parts = rel_path.parts
        
        if "core" in parts:
            self.layer = "core"
        elif "interfaces" in parts:
            self.layer = "interfaces"
        elif "strategies" in parts:
            self.layer = "strategies"
        elif "adapters" in parts:
            self.layer = "adapters"
        elif "processing" in parts:
            self.layer = "processing"
        elif "runtime" in parts:
            self.layer = "runtime"
        else:
            self.layer = "other"

    def visit_Import(self, node: ast.Import) -> None:
        """Check regular imports."""
        for alias in node.names:
            self._check_import(alias.name, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check from imports."""
        if node.module:
            self._check_import(node.module, node.lineno)
        self.generic_visit(node)

    def _check_import(self, module: str, lineno: int) -> None:
        """Check if import violates architectural boundaries."""
        if not module.startswith("fathom."):
            return  # External imports are fine
        
        # Rule 1: Core layer cannot import from adapters
        if self.layer == "core" and module.startswith("fathom.adapters"):
            self.violations.append((
                lineno,
                f"Core layer cannot import from adapters: {module}"
            ))
        
        # Rule 2: Interfaces layer can only import from schemas (and itself)
        if self.layer == "interfaces":
            if not (module.startswith("fathom.schemas") or 
                    module.startswith("fathom.constants") or
                    module.startswith("fathom.interfaces")):
                self.violations.append((
                    lineno,
                    f"Interfaces layer can only import from schemas/constants/interfaces: {module}"
                ))
        
        # Rule 3: Strategies should not import adapters directly
        if self.layer == "strategies":
            if (module.startswith("fathom.adapters.device") or
                module.startswith("fathom.adapters.llm") or
                module.startswith("fathom.adapters.memory") or
                module.startswith("fathom.adapters.knowledge") or
                module.startswith("fathom.adapters.signal") or
                module.startswith("fathom.adapters.storage") or
                module.startswith("fathom.adapters.telemetry")):
                # Exception: vision adapters are allowed (they bridge old/new)
                if not module.startswith("fathom.adapters.vision"):
                    self.violations.append((
                        lineno,
                        f"Strategies should receive ports via dependency injection: {module}"
                    ))
        
        # Rule 4: Processing can only import from schemas, utils, and itself
        if self.layer == "processing":
            if not (module.startswith("fathom.schemas") or
                    module.startswith("fathom.constants") or
                    module.startswith("fathom.utils") or
                    module.startswith("fathom.processing")):
                self.violations.append((
                    lineno,
                    f"Processing layer can only import from schemas/constants/utils/processing: {module}"
                ))


def check_file(file_path: Path, src_root: Path) -> List[Tuple[int, str]]:
    """Check a single Python file for architectural violations."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        
        checker = ImportChecker(file_path, src_root)
        checker.visit(tree)
        return checker.violations
    except SyntaxError:
        return []  # Skip files with syntax errors


def main() -> int:
    """Main entry point."""
    src_root = Path(__file__).parent.parent.parent / "src" / "fathom"
    
    if not src_root.exists():
        print(f"Error: Source root not found: {src_root}")
        return 1
    
    # Find all Python files
    python_files = list(src_root.rglob("*.py"))
    
    total_violations = 0
    files_with_violations: Set[Path] = set()
    
    for file_path in python_files:
        violations = check_file(file_path, src_root)
        if violations:
            files_with_violations.add(file_path)
            total_violations += len(violations)
            
            rel_path = file_path.relative_to(src_root.parent)
            print(f"\n{rel_path}:")
            for lineno, message in violations:
                print(f"  Line {lineno}: {message}")
    
    if total_violations > 0:
        print(f"\n❌ Found {total_violations} architectural violations in {len(files_with_violations)} files")
        return 1
    else:
        print("✅ No architectural violations found")
        return 0


if __name__ == "__main__":
    sys.exit(main())
