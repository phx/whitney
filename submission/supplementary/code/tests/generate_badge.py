"""Generate coverage badge for README."""

import json
import subprocess
from pathlib import Path

def get_coverage():
    """Run coverage and extract percentage."""
    subprocess.run(['pytest', '--cov=core', '--cov-report=json'])
    
    with open('coverage.json') as f:
        data = json.load(f)
        total = data['totals']['percent_covered_display']
    
    return float(total)

def generate_badge(coverage):
    """Generate badge markdown."""
    color = 'red' if coverage < 60 else 'yellow' if coverage < 80 else 'green'
    
    badge = (
        f'![Coverage](https://img.shields.io/badge/coverage-{coverage:.1f}%25-{color})'
    )
    
    readme = Path('../README.md')
    content = readme.read_text()
    
    # Update or add coverage badge
    if '![Coverage]' in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '![Coverage]' in line:
                lines[i] = badge
        content = '\n'.join(lines)
    else:
        content = badge + '\n\n' + content
    
    readme.write_text(content)

if __name__ == '__main__':
    coverage = get_coverage()
    generate_badge(coverage) 