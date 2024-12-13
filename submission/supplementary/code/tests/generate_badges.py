"""Generate coverage badges for documentation."""

import json
import sys
from pathlib import Path

def generate_coverage_badge(coverage_file: Path, output_dir: Path):
    """Generate coverage badge from coverage data."""
    with open(coverage_file) as f:
        data = json.load(f)
        
    total = data["totals"]["percent_covered"]
    color = "red" if total < 80 else "yellow" if total < 90 else "green"
    
    badge = {
        "schemaVersion": 1,
        "label": "coverage",
        "message": f"{total:.1f}%",
        "color": color
    }
    
    output_file = output_dir / "coverage-badge.json"
    with open(output_file, "w") as f:
        json.dump(badge, f)

if __name__ == "__main__":
    coverage_file = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    generate_coverage_badge(coverage_file, output_dir) 