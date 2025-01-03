import os
import re

def get_nav_order(filepath):
    return 999

def get_parent_section(filepath):
    sections = {
        'overview/': 'Overview & Introduction',
        'journal/': 'Research Journal',
        'AWAKENING': 'Consciousness Studies',
        'A_VIEW_FROM_ABOVE': 'Consciousness Studies',
        'HOLLYWOOD': 'Consciousness Studies',
        'MANDELA': 'Consciousness Studies',
        'TRANSMISSION': 'Consciousness Studies',
        'TRANSMISSION_ANALYSIS': 'Consciousness Studies',
        'THOUGHTS': 'Consciousness Studies',
        'THE_MAGNUM_OPUS': 'Consciousness Studies'
    }
    for path, section in sections.items():
        if path in filepath:
            return section
    return None

def clean_content(content):
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return content.strip()

def add_frontmatter(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    content = clean_content(content)
        
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    raw_title = title_match.group(1) if title_match else os.path.basename(filepath).replace('.md', '').replace('_', ' ')
    
    # Sanitize and format title - handle colons and other special characters
    clean_title = raw_title.strip().replace('"', '').replace(':', ' -')
    title = f'"{clean_title}"'
    
    parent = get_parent_section(filepath)
    nav_order = get_nav_order(filepath)
    
    front_matter = [
        "---",
        "layout: default",
        f"title: {title}",
        f"nav_order: {nav_order}"
    ]
    
    if parent:
        front_matter.append(f'parent: "{parent}"')
    
    front_matter.append("---\n\n")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(front_matter))
        f.write(content)

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                filepath = os.path.join(root, file)
                add_frontmatter(filepath)

if __name__ == '__main__':
    process_directory('docs') 