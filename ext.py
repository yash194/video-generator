"""
Simple PDF Extractor using PyMuPDF + VLM (Gemini/OpenAI)

Extracts PDF content in reading order, using VLM to describe 
images/tables/diagrams inline. Outputs a clean text file for LLM consumption.

Dependencies:
    pip install pymupdf pillow google-generativeai openai
"""

import os
import base64
from pathlib import Path
from io import BytesIO

import fitz  # PyMuPDF
from PIL import Image

# VLM providers
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class SimplePDFExtractor:
    """
    Simple PDF extractor that reads content in order and uses VLM
    to describe visual elements (images, diagrams, tables).
    """
    
    def __init__(
        self, 
        provider: str = "auto",  # "gemini", "openai", or "auto"
        api_key: str = None
    ):
        """
        Initialize extractor.
        
        Args:
            provider: VLM provider - "gemini", "openai", or "auto" (detect from env)
            api_key: API key (or set via environment variable)
        """
        self.provider = None
        self.vlm = None
        self.client = None
        
        # Auto-detect provider from environment
        if provider == "auto":
            if os.environ.get("OPENAI_API_KEY"):
                provider = "openai"
            elif os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
                provider = "gemini"
        
        # Initialize Gemini
        if provider == "gemini":
            gemini_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if gemini_key and GEMINI_AVAILABLE:
                genai.configure(api_key=gemini_key)
                self.vlm = genai.GenerativeModel("gemini-1.5-flash")
                self.provider = "gemini"
                print("✓ VLM initialized (Gemini)")
            else:
                print("⚠ Gemini not available - check API key or install google-generativeai")
        
        # Initialize OpenAI
        elif provider == "openai":
            openai_key = api_key or os.environ.get("OPENAI_API_KEY")
            if openai_key and OPENAI_AVAILABLE:
                self.client = openai.OpenAI(api_key=openai_key)
                self.provider = "openai"
                print("✓ VLM initialized (OpenAI GPT-4o)")
            else:
                print("⚠ OpenAI not available - check API key or install openai")
        
        if not self.provider:
            print("⚠ No VLM configured - images will be marked but not described")
    
    def describe_image(self, image_bytes: bytes) -> str:
        """Send image to VLM and get description."""
        if not self.provider:
            return "[Image - VLM not available for description]"
        
        prompt = (
            "Describe this image/diagram/table in detail. "
            "If it's a table, reproduce the data. "
            "If it's a diagram, explain what it shows. "
            "If it's a figure, describe the visual content. "
            "Be concise but complete."
        )
        
        try:
            b64 = base64.b64encode(image_bytes).decode("utf-8")
            
            if self.provider == "gemini":
                image_part = {"mime_type": "image/png", "data": b64}
                response = self.vlm.generate_content([prompt, image_part])
                return response.text.strip()
                
            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{b64}"
                            }}
                        ]
                    }],
                    max_tokens=500
                )
                return response.choices[0].message.content.strip()
                
        except Exception as e:
            return f"[Image description failed: {e}]"
    
    def extract(self, pdf_path: str) -> str:
        """
        Extract PDF content with images described inline.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Complete text content with image descriptions in place
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        output_parts = []
        
        # Add document title
        meta = doc.metadata
        title = meta.get("title") or pdf_path.stem
        output_parts.append(f"# {title}\n")
        
        print(f"Processing {len(doc)} pages...")
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            print(f"  Page {page_num + 1}/{len(doc)}...", end=" ")
            
            # Get all blocks (text and images) in reading order
            blocks = page.get_text("dict")["blocks"]
            blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
            
            page_content = []
            images_described = 0
            described_xrefs = set()
            
            for block in blocks:
                if block["type"] == 0:  # Text block
                    text = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text += span.get("text", "")
                        text += "\n"
                    
                    text = text.strip()
                    if text:
                        page_content.append(text)
                        
                elif block["type"] == 1:  # Image block
                    try:
                        img_rect = fitz.Rect(block["bbox"])
                        clip = page.get_pixmap(clip=img_rect, dpi=150)
                        img_bytes = clip.tobytes("png")
                        
                        if clip.width > 50 and clip.height > 50:
                            description = self.describe_image(img_bytes)
                            page_content.append(f"\n---\n**[Figure/Diagram]**\n{description}\n---\n")
                            images_described += 1
                    except Exception as e:
                        page_content.append(f"\n[Image extraction failed: {e}]\n")
            
            # Check embedded images not in blocks
            # image_list = page.get_images(full=True)
            # for img_info in image_list:
            #     xref = img_info[0]
            #     if xref in described_xrefs:
            #         continue
                    
            #     try:
            #         base_image = doc.extract_image(xref)
            #         if base_image:
            #             img_bytes = base_image["image"]
            #             width, height = base_image["width"], base_image["height"]
                        
            #             if width > 100 and height > 100:
            #                 pil_img = Image.open(BytesIO(img_bytes))
            #                 if pil_img.mode in ("CMYK", "RGBA", "P"):
            #                     pil_img = pil_img.convert("RGB")
                            
            #                 buffer = BytesIO()
            #                 pil_img.save(buffer, format="PNG")
            #                 img_bytes = buffer.getvalue()
                            
            #                 description = self.describe_image(img_bytes)
            #                 page_content.append(f"\n---\n**[Embedded Image]**\n{description}\n---\n")
            #                 images_described += 1
            #                 described_xrefs.add(xref)
            #     except:
            #         pass
            
            print(f"{images_described} images described")
            
            if page_content:
                output_parts.append(f"\n## Page {page_num + 1}\n")
                output_parts.append("\n".join(page_content))
        
        doc.close()
        
        # Clean up
        import re
        full_text = "\n\n".join(output_parts)
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        
        return full_text
    
    def extract_to_file(self, pdf_path: str, output_path: str = None) -> str:
        """Extract PDF and save to text file."""
        pdf_path = Path(pdf_path)
        output_path = output_path or pdf_path.with_suffix(".txt")
        
        content = self.extract(str(pdf_path))
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"\n✓ Saved to: {output_path}")
        return str(output_path)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract PDF content with VLM image descriptions"
    )
    parser.add_argument("pdf_path", help="Path to PDF file")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument(
        "--provider", 
        choices=["gemini", "openai", "auto"],
        default="auto",
        help="VLM provider (default: auto-detect from env)"
    )
    parser.add_argument("--api-key", help="API key (or set via environment)")
    
    args = parser.parse_args()
    
    extractor = SimplePDFExtractor(provider=args.provider, api_key=args.api_key)
    extractor.extract_to_file(args.pdf_path, args.output)


if __name__ == "__main__":
    main()
