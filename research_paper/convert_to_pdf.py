"""
Convert research_paper.md to PDF with LaTeX formulas and embedded images.

Usage: python convert_to_pdf.py

This script:
1. Converts markdown to HTML with MathJax support for LaTeX formulas
2. Embeds all images as base64 data URIs
3. Generates PDF using Chrome/Edge headless mode
4. Opens HTML in browser for manual PDF generation (for best formula rendering)
"""
import subprocess
import sys
import os
import re
import base64
from pathlib import Path

def install_packages():
    """Install required markdown package"""
    try:
        import markdown
    except ImportError:
        print("Installing markdown package...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'markdown', '-q'])
        print("✓ Package installed")

def find_chrome():
    """Find Chrome or Edge browser on Windows"""
    paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    return next((p for p in paths if os.path.exists(p)), None)

def extract_yaml_metadata(content):
    """Extract YAML frontmatter from markdown"""
    yaml_pattern = r'^---\s*\n(.*?)\n---\s*\n'
    match = re.match(yaml_pattern, content, re.DOTALL)

    if not match:
        return {}, content

    metadata = {}
    yaml_content = match.group(1)
    content = content[match.end():]  # Remove YAML from content

    # Parse key fields
    for key in ['title', 'corresponding_author', 'date', 'repository', 'keywords']:
        pattern = rf'{key}:\s*"?([^"\n]+)"?'
        m = re.search(pattern, yaml_content)
        if m:
            metadata[key] = m.group(1).strip()

    # Extract abstract
    abstract_match = re.search(r'abstract:\s*>\s*\n((?:\s+.+\n?)+)', yaml_content)
    if abstract_match:
        abstract = abstract_match.group(1).strip()
        metadata['abstract'] = re.sub(r'\n\s+', ' ', abstract)

    return metadata, content

def embed_images(md_content, base_dir):
    """Replace image references with base64 data URIs"""
    def replace_image(match):
        alt_text = match.group(1)
        img_path = match.group(2)
        full_path = base_dir / img_path

        if full_path.exists():
            try:
                with open(full_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                ext = full_path.suffix.lower()
                mime = 'image/png' if ext == '.png' else 'image/jpeg'
                print(f"  ✓ {img_path}")
                return f'![{alt_text}](data:{mime};base64,{img_data})'
            except:
                pass
        return match.group(0)

    return re.sub(r'!\[(.*?)\]\((.*?)\)', replace_image, md_content)

def create_html(metadata, html_body):
    """Create complete HTML document with styling and MathJax"""
    title_page = ""
    if metadata:
        title_page = f'''<div class="title-page">
<h1 class="paper-title">{metadata.get("title", "")}</h1>
<div class="authors">
<p><strong>Long Hai Huynh</strong><br>
Columbian College of Arts & Sciences, George Washington University<br>
{metadata.get("corresponding_author", "")}</p>
</div>
<p class="date">{metadata.get("date", "")}</p>
<div class="abstract">
<h2>Abstract</h2>
<p>{metadata.get("abstract", "")}</p>
</div>
<p class="keywords"><strong>Keywords:</strong> {metadata.get("keywords", "").strip("[]").replace('"', "")}</p>
<p class="repository"><strong>Repository:</strong> <a href="{metadata.get("repository", "")}">{metadata.get("repository", "")}</a></p>
</div>
<div class="page-break"></div>
'''

    return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Research Paper</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$']],
                displayMath: [['$$', '$$']],
                processEscapes: true
            }},
            startup: {{
                pageReady: () => {{
                    return MathJax.startup.defaultPageReady().then(() => {{
                        console.log('MathJax ready');
                        document.title = 'MATHJAX_READY';
                    }});
                }}
            }}
        }};
    </script>
    <style>
        @media print {{
            @page {{ size: letter; margin: 2.5cm; }}
            h1, h2, h3, h4 {{ page-break-after: avoid; }}
            table, pre, img {{ page-break-inside: avoid; }}
            .page-break {{ page-break-after: always; }}
        }}
        body {{ font-family: Georgia, 'Times New Roman', serif; font-size: 10pt; line-height: 1.4; color: #000; max-width: 900px; margin: 0 auto; padding: 15px; }}
        .title-page {{ text-align: center; padding: 30px 15px; margin-bottom: 15px; }}
        .paper-title {{ font-size: 1.5em; color: #1a1a1a; margin-bottom: 20px; line-height: 1.2; border: none; font-weight: bold; }}
        .authors {{ font-size: 0.95em; margin: 15px 0; }}
        .date {{ font-size: 0.9em; color: #555; margin: 10px 0; }}
        .abstract {{ text-align: left; margin: 25px auto; max-width: 750px; padding: 12px; background: #f8f9fa; border-left: 3px solid #3498db; }}
        .abstract h2 {{ font-size: 1em; margin: 0 0 8px 0; border: none; font-weight: bold; }}
        .abstract p {{ text-align: justify; line-height: 1.4; font-size: 0.9em; }}
        .keywords, .repository {{ text-align: left; margin: 8px auto; max-width: 750px; font-size: 0.85em; }}
        h1 {{ font-size: 1.4em; color: #1a1a1a; border-bottom: 2px solid #3498db; padding-bottom: 6px; margin: 18px 0 10px 0; font-weight: bold; }}
        h2 {{ font-size: 1.2em; color: #2a2a2a; border-bottom: 1px solid #95a5a6; padding-bottom: 4px; margin: 15px 0 8px 0; font-weight: bold; }}
        h3 {{ font-size: 1.1em; color: #3a3a3a; margin: 12px 0 6px 0; font-weight: bold; }}
        h4 {{ font-size: 1.05em; color: #4a4a4a; margin: 10px 0 5px 0; font-weight: bold; }}
        p {{ margin: 6px 0; text-align: justify; line-height: 1.4; }}
        em {{ font-style: italic; }}
        p:has(> em:only-child), img + p {{ text-align: center; font-size: 0.9em; color: #555; margin: 8px 0; }}
        strong {{ font-weight: bold; }}
        ul, ol {{ margin: 6px 0; padding-left: 25px; }}
        li {{ margin: 3px 0; line-height: 1.4; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.8em; }}
        th, td {{ border: 1px solid #ccc; padding: 5px 6px; text-align: left; }}
        th {{ background-color: #3498db; color: white; font-weight: bold; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        code {{ background-color: #f4f4f4; padding: 1px 3px; border-radius: 2px; font-family: 'Courier New', monospace; font-size: 0.85em; }}
        pre {{ background-color: #f4f4f4; padding: 8px; border-radius: 3px; overflow-x: auto; font-size: 0.8em; line-height: 1.3; margin: 8px 0; }}
        pre code {{ background-color: transparent; padding: 0; }}
        blockquote {{ border-left: 3px solid #3498db; padding-left: 10px; margin: 8px 0 8px 0; color: #555; font-style: italic; }}
        a {{ color: #3498db; text-decoration: none; }}
        img {{ max-width: 100%; height: auto; display: block; margin: 12px auto; border: 1px solid #ddd; padding: 3px; background: white; }}
        .MathJax {{ font-size: 1em !important; }}
    </style>
</head>
<body>
{title_page}{html_body}
</body>
</html>'''

def convert_to_pdf(input_md="research_paper.md", output_html="research_paper_with_formulas.html", output_pdf="research_paper.pdf"):
    """Main conversion function"""
    print("="*70)
    print("MARKDOWN TO PDF CONVERTER")
    print("="*70)

    # Install dependencies
    install_packages()
    import markdown

    # Read markdown
    print(f"\nReading: {input_md}")
    with open(input_md, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract metadata
    print("Parsing frontmatter...")
    metadata, content = extract_yaml_metadata(content)
    if metadata:
        print(f"  ✓ Extracted: {', '.join(metadata.keys())}")

    # Embed images
    print("Embedding images...")
    base_dir = Path(input_md).parent.parent
    content = embed_images(content, base_dir)

    # Convert markdown to HTML
    print("Converting to HTML...")
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc', 'footnotes'])
    html_body = md.convert(content)

    # Create complete HTML
    full_html = create_html(metadata, html_body)

    # Write HTML file
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"✓ HTML created: {output_html}")

    # Try automated PDF generation
    chrome = find_chrome()
    if chrome:
        print(f"\nGenerating PDF with {Path(chrome).name}...")
        try:
            cmd = [
                chrome, '--headless=new', '--disable-gpu',
                '--no-pdf-header-footer',
                f'--print-to-pdf={output_pdf}',
                '--virtual-time-budget=15000',
                os.path.abspath(output_html)
            ]
            subprocess.run(cmd, capture_output=True, timeout=45)

            if os.path.exists(output_pdf):
                size = os.path.getsize(output_pdf) / 1024
                print(f"✓ PDF created: {output_pdf} ({size:.1f} KB)")
            else:
                print("✗ Automated PDF failed")
        except:
            print("✗ Automated PDF failed")

    # Open in browser for manual conversion
    print("\n" + "="*70)
    print("MANUAL CONVERSION (RECOMMENDED FOR BEST RESULTS)")
    print("="*70)
    print(f"\nHTML file: {os.path.abspath(output_html)}")
    print("\nFor perfect formula rendering:")
    print("  1. Opening HTML in your browser...")
    print("  2. Wait 3-5 seconds for formulas to render")
    print("  3. Press Ctrl+P")
    print("  4. Enable 'Background graphics'")
    print("  5. Save as PDF")
    print("="*70)

    # Open browser
    try:
        import webbrowser
        webbrowser.open(f'file://{os.path.abspath(output_html)}')
        print("\n✓ Browser opened")
    except:
        print("\n✗ Could not open browser automatically")

if __name__ == "__main__":
    convert_to_pdf()

