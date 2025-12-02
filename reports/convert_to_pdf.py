"""
Convert Final_Report.md to PDF with embedded images.

Usage: python convert_to_pdf.py

This script:
1. Converts markdown to HTML with proper academic formatting
2. Embeds all images as base64 data URIs
3. Generates PDF using Chrome/Edge headless mode
4. Opens HTML in browser for manual PDF generation (recommended)
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
        print(f"  ✗ Not found: {img_path}")
        return match.group(0)

    return re.sub(r'!\[(.*?)\]\((.*?)\)', replace_image, md_content)

def create_html(html_body):
    """Create complete HTML document with styling"""
    return f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>CIB Mango Tree - Final Report</title>
    <style>
        @media print {{
            @page {{ size: letter; margin: 2.5cm; }}
            h1, h2, h3, h4 {{ page-break-after: avoid; }}
            table, pre, img {{ page-break-inside: avoid; }}
        }}
        body {{ 
            font-family: Georgia, 'Times New Roman', serif; 
            font-size: 10pt; 
            line-height: 1.4; 
            color: #000; 
            max-width: 900px; 
            margin: 0 auto; 
            padding: 15px; 
        }}
        
        /* Report title styling */
        h1:first-of-type {{
            text-align: center;
            font-size: 1.3em;
            margin-top: 10px;
            margin-bottom: 5px;
            color: #1a1a1a;
            border: none;
        }}
        h2:nth-of-type(1) {{
            text-align: center;
            font-size: 1.1em;
            margin-top: 5px;
            margin-bottom: 20px;
            color: #333;
            border: none;
            font-weight: normal;
        }}
        
        /* Main title */
        h1:nth-of-type(2) {{
            text-align: center;
            font-size: 1.6em;
            color: #1a1a1a;
            margin-top: 25px;
            margin-bottom: 10px;
            border: none;
            font-weight: bold;
        }}
        
        /* Center only the first few paragraphs (author info) after main title */
        body > p:nth-of-type(1),
        body > p:nth-of-type(2),
        body > p:nth-of-type(3) {{
            text-align: center;
            margin: 8px 0;
        }}
        
        h1 {{ 
            font-size: 1.4em; 
            color: #1a1a1a; 
            border-bottom: 2px solid #3498db; 
            padding-bottom: 6px; 
            margin: 18px 0 10px 0; 
            font-weight: bold; 
        }}
        h2 {{ 
            font-size: 1.2em; 
            color: #2a2a2a; 
            border-bottom: 1px solid #95a5a6; 
            padding-bottom: 4px; 
            margin: 15px 0 8px 0; 
            font-weight: bold; 
        }}
        h3 {{ 
            font-size: 1.1em; 
            color: #3a3a3a; 
            margin: 12px 0 6px 0; 
            font-weight: bold; 
        }}
        h4 {{ 
            font-size: 1.05em; 
            color: #4a4a4a; 
            margin: 10px 0 5px 0; 
            font-weight: bold; 
        }}
        
        p {{ 
            margin: 6px 0; 
            text-align: justify; 
            line-height: 1.4; 
        }}
        
        em {{ 
            font-style: italic; 
        }}
        
        /* Figure captions */
        p:has(> em:only-child), img + p {{ 
            text-align: center; 
            font-size: 0.9em; 
            color: #555; 
            margin: 8px 0; 
        }}
        
        strong {{ 
            font-weight: bold; 
        }}
        
        ul, ol {{ 
            margin: 6px 0; 
            padding-left: 25px; 
        }}
        li {{ 
            margin: 3px 0; 
            line-height: 1.4; 
        }}
        
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
            margin: 10px 0; 
            font-size: 0.8em; 
        }}
        th, td {{ 
            border: 1px solid #ccc; 
            padding: 5px 6px; 
            text-align: left; 
        }}
        th {{ 
            background-color: #3498db; 
            color: white; 
            font-weight: bold; 
        }}
        tr:nth-child(even) {{ 
            background-color: #f9f9f9; 
        }}
        
        code {{ 
            background-color: #f4f4f4; 
            padding: 1px 3px; 
            border-radius: 2px; 
            font-family: 'Courier New', monospace; 
            font-size: 0.85em; 
        }}
        pre {{ 
            background-color: #f4f4f4; 
            padding: 8px; 
            border-radius: 3px; 
            overflow-x: auto; 
            font-size: 0.75em; 
            line-height: 1.3; 
            margin: 8px 0; 
        }}
        pre code {{ 
            background-color: transparent; 
            padding: 0; 
        }}
        
        blockquote {{ 
            border-left: 3px solid #3498db; 
            padding-left: 10px; 
            margin: 8px 0 8px 0; 
            color: #555; 
            font-style: italic; 
        }}
        
        a {{ 
            color: #3498db; 
            text-decoration: none; 
        }}
        
        img {{ 
            max-width: 100%; 
            height: auto; 
            display: block; 
            margin: 12px auto; 
            border: 1px solid #ddd; 
            padding: 3px; 
            background: white; 
        }}
        
        hr {{
            border: none;
            border-top: 1px solid #ccc;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>'''

def convert_to_pdf(input_md="Final_Report.md", output_html="Final_Report.html", output_pdf="Final_Report.pdf"):
    """Main conversion function"""
    print("="*70)
    print("FINAL REPORT: MARKDOWN TO PDF CONVERTER")
    print("="*70)

    # Install dependencies
    install_packages()
    import markdown

    # Read markdown
    print(f"\nReading: {input_md}")
    with open(input_md, 'r', encoding='utf-8') as f:
        content = f.read()

    # Embed images
    print("Embedding images...")
    base_dir = Path(input_md).parent.parent
    content = embed_images(content, base_dir)

    # Convert markdown to HTML
    print("Converting to HTML...")
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc', 'footnotes'])
    html_body = md.convert(content)

    # Create complete HTML
    full_html = create_html(html_body)

    # Write HTML file
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(full_html)
    print(f"✓ HTML created: {output_html}")

    # Try automated PDF generation
    chrome = find_chrome()
    if chrome:
        print(f"\nGenerating PDF with {Path(chrome).name}...")
        try:
            output_pdf_abs = os.path.abspath(output_pdf)
            cmd = [
                chrome,
                '--headless=new',
                '--disable-gpu',
                '--no-pdf-header-footer',
                f'--print-to-pdf={output_pdf_abs}',
                '--virtual-time-budget=10000',
                '--run-all-compositor-stages-before-draw',
                f'file:///{os.path.abspath(output_html)}'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)

            # Give it a moment to finish writing
            import time
            time.sleep(1)

            if os.path.exists(output_pdf):
                size = os.path.getsize(output_pdf) / 1024
                print(f"✓ PDF created: {output_pdf} ({size:.1f} KB)")
            else:
                print("✗ Automated PDF failed")
                if result.stderr:
                    print(f"  Error: {result.stderr[:200]}")
        except Exception as e:
            print(f"✗ Automated PDF failed: {e}")

    # Open in browser for manual conversion
    print("\n" + "="*70)
    print("MANUAL CONVERSION (RECOMMENDED)")
    print("="*70)
    print(f"\nHTML file: {os.path.abspath(output_html)}")
    print("\nFor best results:")
    print("  1. Opening HTML in your browser...")
    print("  2. Press Ctrl+P")
    print("  3. Enable 'Background graphics'")
    print("  4. Save as PDF")
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

