"""
PDF Content Extractor for Language Models - Scene Generation Ready

This module extracts text, images, tables, and other content from PDF files
and stores them in a structured format (JSON/Markdown) that can be easily
consumed by language models for scene generation and direction following.

Key Features:
- VLM integration for image scene descriptions (Gemini/GPT-4o/LLaVA)
- Smart layout preservation (filters headers/footers)
- Inline image markers (maintains spatial context)
- Cross-page paragraph continuity

Dependencies:
    pip install pymupdf pillow pytesseract pdfplumber google-generativeai openai

Optional for OCR:
    brew install tesseract  # macOS
    apt-get install tesseract-ocr  # Ubuntu/Debian
"""

import os
import json
import base64
import hashlib
import re
from pathlib import Path
from typing import Optional, Union, Callable
from datetime import datetime
from dataclasses import dataclass, field, asdict
from io import BytesIO

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    print("Warning: PyMuPDF not installed. Run: pip install pymupdf")

try:
    from PIL import Image
except ImportError:
    Image = None
    print("Warning: Pillow not installed. Run: pip install pillow")

try:
    import pdfplumber
except ImportError:
    pdfplumber = None
    print("Warning: pdfplumber not installed. Run: pip install pdfplumber")

try:
    import pytesseract
except ImportError:
    pytesseract = None
    print("Warning: pytesseract not installed. Run: pip install pytesseract")

# VLM providers (optional)
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


@dataclass
class ExtractedImage:
    """Represents an extracted image from the PDF."""
    page_number: int
    image_index: int
    block_index: int  # Position in content flow
    width: int
    height: int
    format: str
    base64_data: str  # Base64 encoded image data
    file_path: Optional[str] = None  # Path if saved to disk
    ocr_text: Optional[str] = None  # OCR extracted text from image
    vlm_description: Optional[str] = None  # VLM scene description
    caption: Optional[str] = None  # Detected caption near image


@dataclass
class ExtractedTable:
    """Represents an extracted table from the PDF."""
    page_number: int
    table_index: int
    block_index: int  # Position in content flow
    rows: list
    markdown: str  # Table in markdown format
    csv: str  # Table in CSV format


@dataclass 
class ContentBlock:
    """Represents a block of content (text, image marker, or table) in reading order."""
    block_type: str  # 'text', 'image', 'table'
    content: str  # Text content or reference marker
    page_number: int
    block_index: int
    y_position: float  # Vertical position for sorting
    image_ref: Optional[str] = None  # Image ID if type is 'image'
    table_ref: Optional[int] = None  # Table index if type is 'table'


@dataclass
class ExtractedPage:
    """Represents extracted content from a single PDF page."""
    page_number: int
    text: str  # Clean text (legacy, for backward compatibility)
    content_flow: list = field(default_factory=list)  # Ordered ContentBlocks
    images: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    links: list = field(default_factory=list)
    annotations: list = field(default_factory=list)


@dataclass
class PDFMetadata:
    """PDF document metadata."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    page_count: int = 0
    file_size: int = 0
    file_name: str = ""


@dataclass
class ExtractedPDF:
    """Complete extracted content from a PDF document."""
    metadata: PDFMetadata
    pages: list = field(default_factory=list)
    extraction_date: str = field(default_factory=lambda: datetime.now().isoformat())
    extraction_settings: dict = field(default_factory=dict)


class VLMDescriber:
    """
    Handles image description using Vision Language Models.
    Supports Gemini, GPT-4o, and custom VLM callbacks.
    """
    
    def __init__(
        self,
        provider: str = "gemini",  # 'gemini', 'openai', 'custom'
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        custom_callback: Optional[Callable] = None,
        prompt_template: Optional[str] = None,
    ):
        """
        Initialize VLM describer.
        
        Args:
            provider: VLM provider ('gemini', 'openai', 'custom', 'none')
            api_key: API key (or set via environment variable)
            model: Model name (defaults based on provider)
            custom_callback: Custom function(base64_image) -> description
            prompt_template: Custom prompt for scene description
        """
        self.provider = provider.lower()
        self.custom_callback = custom_callback
        self.prompt = prompt_template or (
            "Describe this image in detail for a scene generation prompt. "
            "Focus on: 1) Main subjects and their actions, 2) Setting and environment, "
            "3) Lighting conditions, 4) Camera angle and perspective, "
            "5) Any text or labels visible. Be specific and visual."
        )
        
        if self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")
            api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model or "gemini-1.5-flash")
            else:
                print("Warning: No Gemini API key found. VLM descriptions will be disabled.")
                self.provider = "none"
                
        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai not installed. Run: pip install openai")
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
                self.model_name = model or "gpt-4o"
            else:
                print("Warning: No OpenAI API key found. VLM descriptions will be disabled.")
                self.provider = "none"
                
        elif self.provider == "custom":
            if not custom_callback:
                raise ValueError("custom_callback required for custom provider")
                
    def describe(self, base64_image: str, image_format: str = "png") -> Optional[str]:
        """
        Generate a scene description for an image.
        
        Args:
            base64_image: Base64 encoded image data
            image_format: Image format (png, jpg, webp)
            
        Returns:
            Scene description string or None if failed
        """
        if self.provider == "none":
            return None
            
        try:
            if self.provider == "gemini":
                return self._describe_gemini(base64_image, image_format)
            elif self.provider == "openai":
                return self._describe_openai(base64_image, image_format)
            elif self.provider == "custom":
                return self.custom_callback(base64_image)
        except Exception as e:
            print(f"VLM description failed: {e}")
            return None
            
    def _describe_gemini(self, base64_image: str, image_format: str) -> str:
        """Generate description using Gemini."""
        image_bytes = base64.b64decode(base64_image)
        
        # Create image part for Gemini
        image_part = {
            "mime_type": f"image/{image_format}",
            "data": base64_image
        }
        
        response = self.model.generate_content([self.prompt, image_part])
        return response.text.strip()
        
    def _describe_openai(self, base64_image: str, image_format: str) -> str:
        """Generate description using GPT-4o."""
        mime_type = f"image/{image_format}"
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()


class PDFExtractor:
    """
    Comprehensive PDF content extractor that extracts text, images, tables,
    and annotations from PDF files for language model consumption.
    
    Scene Generation Ready Features:
    - VLM integration for visual scene descriptions
    - Smart layout preservation (filters headers/footers)
    - Inline image markers (maintains spatial context)
    - Cross-page paragraph continuity
    """
    
    def __init__(
        self,
        extract_images: bool = True,
        extract_tables: bool = True,
        perform_ocr: bool = True,
        use_vlm: bool = False,
        vlm_provider: str = "gemini",
        vlm_api_key: Optional[str] = None,
        vlm_model: Optional[str] = None,
        vlm_callback: Optional[Callable] = None,
        save_images: bool = True,
        output_dir: Optional[str] = None,
        image_format: str = "png",
        min_image_size: int = 50,  # Minimum width/height to extract
        image_quality: int = 95,
        dpi: int = 150,  # DPI for image extraction
        filter_headers_footers: bool = True,
        header_footer_margin: float = 0.05,  # 5% of page height
        preserve_reading_order: bool = True,
        inline_image_markers: bool = True,
    ):
        """
        Initialize the PDF extractor.
        
        Args:
            extract_images: Whether to extract images from PDF
            extract_tables: Whether to extract tables from PDF
            perform_ocr: Whether to perform OCR on images
            use_vlm: Whether to use VLM for image descriptions
            vlm_provider: VLM provider ('gemini', 'openai', 'custom', 'none')
            vlm_api_key: API key for VLM (or use environment variable)
            vlm_model: Specific model to use
            vlm_callback: Custom VLM function for 'custom' provider
            save_images: Whether to save images to disk
            output_dir: Directory to save extracted content
            image_format: Format to save images (png, jpg, webp)
            min_image_size: Minimum image dimension to extract
            image_quality: JPEG/WebP quality (1-100)
            dpi: DPI for rasterizing PDF pages
            filter_headers_footers: Remove header/footer content
            header_footer_margin: Margin for header/footer detection (0-0.5)
            preserve_reading_order: Sort content by reading order (top-to-bottom, left-to-right)
            inline_image_markers: Insert image markers in text flow
        """
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.perform_ocr = perform_ocr and pytesseract is not None
        self.use_vlm = use_vlm
        self.save_images = save_images
        self.output_dir = Path(output_dir) if output_dir else None
        self.image_format = image_format.lower()
        self.min_image_size = min_image_size
        self.image_quality = image_quality
        self.dpi = dpi
        self.filter_headers_footers = filter_headers_footers
        self.header_footer_margin = header_footer_margin
        self.preserve_reading_order = preserve_reading_order
        self.inline_image_markers = inline_image_markers
        
        # Initialize VLM describer
        self.vlm = None
        if self.use_vlm:
            try:
                self.vlm = VLMDescriber(
                    provider=vlm_provider,
                    api_key=vlm_api_key,
                    model=vlm_model,
                    custom_callback=vlm_callback,
                )
            except Exception as e:
                print(f"Warning: VLM initialization failed: {e}")
                self.use_vlm = False
        
        self._validate_dependencies()
        
        # Image tracking for inline markers
        self._current_page_images = {}
    
    def _validate_dependencies(self):
        """Check if required dependencies are available."""
        if fitz is None:
            raise ImportError(
                "PyMuPDF is required. Install with: pip install pymupdf"
            )
        if self.extract_tables and pdfplumber is None:
            print("Warning: pdfplumber not available. Table extraction disabled.")
            self.extract_tables = False
        if self.perform_ocr and pytesseract is None:
            print("Warning: pytesseract not available. OCR disabled.")
            self.perform_ocr = False
    
    def extract(self, pdf_path: str) -> ExtractedPDF:
        """
        Extract all content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ExtractedPDF object containing all extracted content
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Setup output directory
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        elif self.save_images:
            self.output_dir = pdf_path.parent / f"{pdf_path.stem}_extracted"
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract metadata
        metadata = self._extract_metadata(pdf_path)
        
        # Extract page content
        pages = []
        doc = fitz.open(pdf_path)
        
        # Also open with pdfplumber for table extraction
        plumber_pdf = None
        if self.extract_tables and pdfplumber:
            plumber_pdf = pdfplumber.open(pdf_path)
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                plumber_page = plumber_pdf.pages[page_num] if plumber_pdf else None
                
                extracted_page = self._extract_page(
                    page, page_num + 1, plumber_page, doc
                )
                pages.append(extracted_page)
        finally:
            doc.close()
            if plumber_pdf:
                plumber_pdf.close()
        
        return ExtractedPDF(
            metadata=metadata,
            pages=pages,
            extraction_settings={
                "extract_images": self.extract_images,
                "extract_tables": self.extract_tables,
                "perform_ocr": self.perform_ocr,
                "use_vlm": self.use_vlm,
                "dpi": self.dpi,
                "min_image_size": self.min_image_size,
                "filter_headers_footers": self.filter_headers_footers,
                "inline_image_markers": self.inline_image_markers,
            }
        )
    
    def _extract_metadata(self, pdf_path: Path) -> PDFMetadata:
        """Extract PDF metadata."""
        doc = fitz.open(pdf_path)
        meta = doc.metadata
        
        metadata = PDFMetadata(
            title=meta.get("title"),
            author=meta.get("author"),
            subject=meta.get("subject"),
            keywords=meta.get("keywords"),
            creator=meta.get("creator"),
            producer=meta.get("producer"),
            creation_date=meta.get("creationDate"),
            modification_date=meta.get("modDate"),
            page_count=len(doc),
            file_size=pdf_path.stat().st_size,
            file_name=pdf_path.name,
        )
        
        doc.close()
        return metadata
    
    def _extract_page(
        self, 
        page: "fitz.Page", 
        page_num: int,
        plumber_page,
        doc: "fitz.Document"
    ) -> ExtractedPage:
        """Extract content from a single page with reading order preservation."""
        page_height = page.rect.height
        page_width = page.rect.width
        
        # Get structured content with positions
        blocks = page.get_text("dict")["blocks"]
        
        content_flow = []
        images = []
        tables = []
        block_index = 0
        
        # Pre-extract images to build reference map
        image_map = {}  # block_number -> image_data
        if self.extract_images:
            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    extracted_img = self._extract_single_image(
                        doc, xref, page_num, img_idx, block_index
                    )
                    if extracted_img:
                        image_map[xref] = extracted_img
                except Exception as e:
                    print(f"Error extracting image {img_idx} from page {page_num}: {e}")
        
        # Process blocks in reading order
        sorted_blocks = sorted(blocks, key=lambda b: (b.get("bbox", [0, 0, 0, 0])[1], b.get("bbox", [0, 0, 0, 0])[0]))
        
        for block in sorted_blocks:
            bbox = block.get("bbox", [0, 0, 0, 0])
            y_pos = bbox[1]
            
            # Filter headers/footers
            if self.filter_headers_footers:
                if y_pos < page_height * self.header_footer_margin:
                    continue  # Skip header
                if bbox[3] > page_height * (1 - self.header_footer_margin):
                    continue  # Skip footer
            
            if block.get("type") == 0:  # Text block
                text = self._extract_text_from_block(block)
                if text.strip():
                    content_flow.append(ContentBlock(
                        block_type="text",
                        content=text.strip(),
                        page_number=page_num,
                        block_index=block_index,
                        y_position=y_pos,
                    ))
                    block_index += 1
                    
            elif block.get("type") == 1:  # Image block
                # Find corresponding extracted image
                if self.inline_image_markers:
                    img_id = f"IMG_P{page_num}_{block.get('number', block_index)}"
                    content_flow.append(ContentBlock(
                        block_type="image",
                        content=f"[SCENE_IMAGE: {img_id}]",
                        page_number=page_num,
                        block_index=block_index,
                        y_position=y_pos,
                        image_ref=img_id,
                    ))
                    block_index += 1
        
        # Add extracted images to the page
        for xref, img_data in image_map.items():
            images.append(img_data)
        
        # Extract tables with positions
        if self.extract_tables and plumber_page:
            tables = self._extract_tables_from_page(plumber_page, page_num)
            # Insert table markers into content flow
            for table in tables:
                # Approximate position (pdfplumber doesn't give exact coords easily)
                content_flow.append(ContentBlock(
                    block_type="table",
                    content=f"[TABLE: Table_{page_num}_{table.table_index}]",
                    page_number=page_num,
                    block_index=block_index,
                    y_position=0,  # Will be at end
                    table_ref=table.table_index,
                ))
                block_index += 1
        
        # Sort content flow by position if needed
        if self.preserve_reading_order:
            content_flow.sort(key=lambda b: (b.y_position, b.block_index))
        
        # Generate legacy text (for backward compatibility)
        legacy_text = "\n\n".join(
            b.content for b in content_flow if b.block_type == "text"
        )
        
        # Extract links
        links = self._extract_links_from_page(page)
        
        # Extract annotations
        annotations = self._extract_annotations_from_page(page)
        
        return ExtractedPage(
            page_number=page_num,
            text=legacy_text,
            content_flow=[asdict(b) for b in content_flow],
            images=images,
            tables=tables,
            links=links,
            annotations=annotations,
        )
    
    def _extract_text_from_block(self, block: dict) -> str:
        """Extract text from a PyMuPDF text block."""
        lines = block.get("lines", [])
        text_parts = []
        
        for line in lines:
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans)
            text_parts.append(line_text)
        
        return " ".join(text_parts)
    
    def _extract_single_image(
        self, 
        doc: "fitz.Document",
        xref: int,
        page_num: int,
        img_index: int,
        block_index: int
    ) -> Optional[ExtractedImage]:
        """Extract a single image from the PDF."""
        try:
            base_image = doc.extract_image(xref)
            if base_image is None:
                return None
            
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            width = base_image["width"]
            height = base_image["height"]
            
            # Skip small images (likely icons or artifacts)
            if width < self.min_image_size or height < self.min_image_size:
                return None
            
            # Convert to PIL Image for processing
            if Image:
                pil_image = Image.open(BytesIO(image_bytes))
                
                # Convert CMYK to RGB
                if pil_image.mode == "CMYK":
                    pil_image = pil_image.convert("RGB")
                # Convert RGBA/P to RGB
                elif pil_image.mode in ("RGBA", "P"):
                    pil_image = pil_image.convert("RGB")
                
                # Re-encode in desired format
                buffer = BytesIO()
                if self.image_format == "jpg":
                    pil_image.save(buffer, format="JPEG", quality=self.image_quality)
                elif self.image_format == "webp":
                    pil_image.save(buffer, format="WEBP", quality=self.image_quality)
                else:
                    pil_image.save(buffer, format="PNG")
                
                image_bytes = buffer.getvalue()
            
            # Encode to base64
            base64_data = base64.b64encode(image_bytes).decode("utf-8")
            
            # Generate hash for filename
            img_hash = hashlib.md5(image_bytes).hexdigest()[:8]
            
            # Save image to disk if requested
            file_path = None
            if self.save_images and self.output_dir:
                filename = f"page{page_num}_img{img_index}_{img_hash}.{self.image_format}"
                file_path = str(self.output_dir / filename)
                with open(file_path, "wb") as f:
                    f.write(image_bytes)
            
            # Perform OCR on image if requested
            ocr_text = None
            if self.perform_ocr and Image and pytesseract:
                try:
                    pil_image = Image.open(BytesIO(image_bytes))
                    ocr_text = pytesseract.image_to_string(pil_image).strip()
                    if not ocr_text:
                        ocr_text = None
                except Exception as e:
                    print(f"OCR failed for image {img_index} on page {page_num}: {e}")
            
            # Get VLM description if enabled
            vlm_description = None
            if self.use_vlm and self.vlm:
                vlm_description = self.vlm.describe(base64_data, self.image_format)
            
            return ExtractedImage(
                page_number=page_num,
                image_index=img_index,
                block_index=block_index,
                width=width,
                height=height,
                format=self.image_format,
                base64_data=base64_data,
                file_path=file_path,
                ocr_text=ocr_text,
                vlm_description=vlm_description,
            )
            
        except Exception as e:
            raise e
    
    def _extract_tables_from_page(self, plumber_page, page_num: int) -> list:
        """Extract tables from a PDF page using pdfplumber."""
        tables = []
        
        try:
            raw_tables = plumber_page.extract_tables()
            
            for table_index, table in enumerate(raw_tables):
                if not table or len(table) == 0:
                    continue
                
                # Clean up table cells
                cleaned_rows = []
                for row in table:
                    cleaned_row = [
                        str(cell).strip() if cell is not None else ""
                        for cell in row
                    ]
                    cleaned_rows.append(cleaned_row)
                
                if not cleaned_rows:
                    continue
                
                # Generate markdown table
                markdown = self._table_to_markdown(cleaned_rows)
                
                # Generate CSV
                csv_content = self._table_to_csv(cleaned_rows)
                
                tables.append(ExtractedTable(
                    page_number=page_num,
                    table_index=table_index,
                    block_index=0,
                    rows=cleaned_rows,
                    markdown=markdown,
                    csv=csv_content,
                ))
                
        except Exception as e:
            print(f"Error extracting tables from page {page_num}: {e}")
        
        return tables
    
    def _table_to_markdown(self, rows: list) -> str:
        """Convert table rows to markdown format."""
        if not rows:
            return ""
        
        lines = []
        
        # Header row
        header = rows[0]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        
        # Data rows
        for row in rows[1:]:
            # Ensure row has same number of columns as header
            while len(row) < len(header):
                row.append("")
            lines.append("| " + " | ".join(row[:len(header)]) + " |")
        
        return "\n".join(lines)
    
    def _table_to_csv(self, rows: list) -> str:
        """Convert table rows to CSV format."""
        lines = []
        for row in rows:
            # Escape quotes and wrap cells containing commas
            escaped = []
            for cell in row:
                if "," in cell or '"' in cell or "\n" in cell:
                    cell = '"' + cell.replace('"', '""') + '"'
                escaped.append(cell)
            lines.append(",".join(escaped))
        return "\n".join(lines)
    
    def _extract_links_from_page(self, page: "fitz.Page") -> list:
        """Extract hyperlinks from a PDF page."""
        links = []
        for link in page.get_links():
            link_info = {
                "type": link.get("kind", 0),
                "uri": link.get("uri"),
                "page": link.get("page"),
            }
            if link_info["uri"] or link_info["page"] is not None:
                links.append(link_info)
        return links
    
    def _extract_annotations_from_page(self, page: "fitz.Page") -> list:
        """Extract annotations (comments, highlights) from a PDF page."""
        annotations = []
        for annot in page.annots():
            if annot:
                annot_info = {
                    "type": annot.type[1],  # Type name
                    "content": annot.info.get("content", ""),
                    "title": annot.info.get("title", ""),
                }
                if annot_info["content"] or annot_info["title"]:
                    annotations.append(annot_info)
        return annotations
    
    def to_json(self, extracted: ExtractedPDF, include_base64: bool = False) -> str:
        """
        Convert extracted PDF content to JSON format.
        
        Args:
            extracted: ExtractedPDF object
            include_base64: Whether to include base64 image data
            
        Returns:
            JSON string
        """
        data = asdict(extracted)
        
        # Optionally remove base64 data to reduce size
        if not include_base64:
            for page in data["pages"]:
                for image in page["images"]:
                    image["base64_data"] = "[BASE64_DATA_OMITTED]"
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    def to_markdown(self, extracted: ExtractedPDF) -> str:
        """
        Convert extracted PDF content to Markdown format optimized for LLMs.
        
        Args:
            extracted: ExtractedPDF object
            
        Returns:
            Markdown string
        """
        lines = []
        
        # Document header with metadata
        meta = extracted.metadata
        lines.append(f"# {meta.title or meta.file_name}")
        lines.append("")
        
        if meta.author:
            lines.append(f"**Author:** {meta.author}")
        if meta.subject:
            lines.append(f"**Subject:** {meta.subject}")
        if meta.keywords:
            lines.append(f"**Keywords:** {meta.keywords}")
        lines.append(f"**Pages:** {meta.page_count}")
        lines.append(f"**Extracted:** {extracted.extraction_date}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Build image/table lookup maps
        image_map = {}
        table_map = {}
        for page in extracted.pages:
            for img in page.images:
                img_id = f"IMG_P{page.page_number}_{img.block_index}"
                image_map[img_id] = img
            for tbl in page.tables:
                table_map[f"Table_{page.page_number}_{tbl.table_index}"] = tbl
        
        # Content by page with inline elements
        for page in extracted.pages:
            lines.append(f"## Page {page.page_number}")
            lines.append("")
            
            # Process content flow
            for block in page.content_flow:
                if block["block_type"] == "text":
                    lines.append(block["content"])
                    lines.append("")
                    
                elif block["block_type"] == "image":
                    img_ref = block.get("image_ref", "")
                    if img_ref in image_map:
                        img = image_map[img_ref]
                        lines.append(f"**[IMAGE: {img_ref}]**")
                        lines.append(f"  - Dimensions: {img.width}x{img.height}")
                        if img.vlm_description:
                            lines.append(f"  - Scene: {img.vlm_description}")
                        elif img.ocr_text:
                            lines.append(f"  - Text: {img.ocr_text}")
                        if img.file_path:
                            lines.append(f"  - File: {img.file_path}")
                        lines.append("")
                        
                elif block["block_type"] == "table":
                    tbl_ref = block.get("table_ref")
                    tbl_key = f"Table_{page.page_number}_{tbl_ref}"
                    if tbl_key in table_map:
                        tbl = table_map[tbl_key]
                        lines.append(f"**[TABLE: {tbl_key}]**")
                        lines.append("")
                        lines.append(tbl.markdown)
                        lines.append("")
            
            # Links
            if page.links:
                lines.append("### Links")
                for link in page.links:
                    if link.get("uri"):
                        lines.append(f"- [{link['uri']}]({link['uri']})")
                    elif link.get("page") is not None:
                        lines.append(f"- Internal link to page {link['page'] + 1}")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)
    
    def to_llm_context(self, extracted: ExtractedPDF, max_tokens: int = None) -> str:
        """
        Convert extracted PDF to a format optimized for LLM context with
        continuous flow and inline scene descriptions.
        
        This format:
        - Maintains spatial relationship between text and images
        - Preserves cross-page paragraph continuity
        - Includes VLM scene descriptions inline
        - Uses structured markers for easy parsing
        
        Args:
            extracted: ExtractedPDF object
            max_tokens: Optional token limit (rough estimate)
            
        Returns:
            LLM-optimized context string
        """
        lines = []
        
        # Compact header
        meta = extracted.metadata
        lines.append(f"<document title=\"{meta.title or meta.file_name}\" pages=\"{meta.page_count}\">")
        lines.append("")
        
        # Build image/table lookup maps
        image_map = {}
        table_map = {}
        for page in extracted.pages:
            for img in page.images:
                img_id = f"IMG_P{page.page_number}_{img.block_index}"
                image_map[img_id] = img
            for tbl in page.tables:
                table_map[f"Table_{page.page_number}_{tbl.table_index}"] = tbl
        
        # Continuous content with paragraph detection
        previous_ended_with_sentence = True
        
        for page_idx, page in enumerate(extracted.pages):
            # Light page marker (not structural break)
            lines.append(f"<!-- page {page.page_number} -->")
            
            for block in page.content_flow:
                if block["block_type"] == "text":
                    text = block["content"]
                    
                    # Handle cross-page continuity
                    if not previous_ended_with_sentence and lines:
                        # Previous block didn't end with punctuation - continue sentence
                        if lines[-1] and not lines[-1].startswith("<") and not lines[-1].startswith("<!--"):
                            lines[-1] = lines[-1] + " " + text
                            continue
                    
                    lines.append(text)
                    
                    # Check if this ends with sentence-ending punctuation
                    previous_ended_with_sentence = bool(
                        text and text.strip() and text.strip()[-1] in ".!?:;"
                    )
                    
                elif block["block_type"] == "image":
                    img_ref = block.get("image_ref", "")
                    if img_ref in image_map:
                        img = image_map[img_ref]
                        
                        # Inline image with description
                        if img.vlm_description:
                            lines.append(f"<scene id=\"{img_ref}\">{img.vlm_description}</scene>")
                        elif img.ocr_text:
                            lines.append(f"<figure id=\"{img_ref}\" text=\"{img.ocr_text}\" />")
                        else:
                            lines.append(f"<figure id=\"{img_ref}\" width=\"{img.width}\" height=\"{img.height}\" />")
                    
                    previous_ended_with_sentence = True
                    
                elif block["block_type"] == "table":
                    tbl_ref = block.get("table_ref")
                    tbl_key = f"Table_{page.page_number}_{tbl_ref}"
                    if tbl_key in table_map:
                        tbl = table_map[tbl_key]
                        lines.append(f"<table id=\"{tbl_key}\">")
                        lines.append(tbl.markdown)
                        lines.append("</table>")
                    
                    previous_ended_with_sentence = True
            
            lines.append("")
        
        lines.append("</document>")
        
        result = "\n".join(lines)
        
        # Rough token limit enforcement (estimate 4 chars per token)
        if max_tokens and len(result) > max_tokens * 4:
            result = result[:max_tokens * 4] + "\n... [TRUNCATED]"
        
        return result
    
    def save(
        self, 
        extracted: ExtractedPDF, 
        output_path: str,
        format: str = "json",
        include_base64: bool = False
    ):
        """
        Save extracted content to a file.
        
        Args:
            extracted: ExtractedPDF object
            output_path: Path to save the output
            format: Output format ('json', 'markdown', 'md', 'llm')
            include_base64: Include base64 image data (only for JSON)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            content = self.to_json(extracted, include_base64=include_base64)
        elif format.lower() in ("markdown", "md"):
            content = self.to_markdown(extracted)
        elif format.lower() == "llm":
            content = self.to_llm_context(extracted)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"Saved extracted content to: {output_path}")


def extract_pdf(
    pdf_path: str,
    output_path: Optional[str] = None,
    output_format: str = "markdown",
    extract_images: bool = True,
    extract_tables: bool = True,
    perform_ocr: bool = True,
    use_vlm: bool = False,
    vlm_provider: str = "gemini",
    save_images: bool = True,
) -> str:
    """
    Convenience function to extract content from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        output_path: Path to save output (default: same as PDF with new extension)
        output_format: Output format ('json', 'markdown', 'llm')
        extract_images: Whether to extract images
        extract_tables: Whether to extract tables
        perform_ocr: Whether to perform OCR on images
        use_vlm: Whether to use VLM for scene descriptions
        vlm_provider: VLM provider ('gemini', 'openai')
        save_images: Whether to save images to disk
        
    Returns:
        Path to the saved output file
    """
    pdf_path = Path(pdf_path)
    
    # Default output path
    if output_path is None:
        ext = ".json" if output_format == "json" else ".md"
        output_path = pdf_path.with_suffix(ext)
    
    # Create extractor
    extractor = PDFExtractor(
        extract_images=extract_images,
        extract_tables=extract_tables,
        perform_ocr=perform_ocr,
        use_vlm=use_vlm,
        vlm_provider=vlm_provider,
        save_images=save_images,
        output_dir=pdf_path.parent / f"{pdf_path.stem}_extracted",
    )
    
    # Extract content
    extracted = extractor.extract(str(pdf_path))
    
    # Save output
    extractor.save(extracted, str(output_path), format=output_format)
    
    return str(output_path)


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract content from PDF files for language model consumption (Scene Generation Ready)"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "-o", "--output", 
        help="Output file path (default: same as input with .md extension)"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["json", "markdown", "md", "llm"],
        default="llm",
        help="Output format (default: llm)"
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip image extraction"
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Skip table extraction"
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Skip OCR on images"
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Don't save images to disk"
    )
    parser.add_argument(
        "--vlm",
        action="store_true",
        help="Enable VLM scene descriptions (requires API key)"
    )
    parser.add_argument(
        "--vlm-provider",
        choices=["gemini", "openai"],
        default="gemini",
        help="VLM provider (default: gemini)"
    )
    parser.add_argument(
        "--no-filter-headers",
        action="store_true",
        help="Don't filter out headers/footers"
    )
    
    args = parser.parse_args()
    
    # Create extractor with all options
    extractor = PDFExtractor(
        extract_images=not args.no_images,
        extract_tables=not args.no_tables,
        perform_ocr=not args.no_ocr,
        use_vlm=args.vlm,
        vlm_provider=args.vlm_provider,
        save_images=not args.no_save_images,
        filter_headers_footers=not args.no_filter_headers,
    )
    
    # Extract content
    pdf_path = Path(args.pdf_path)
    extracted = extractor.extract(str(pdf_path))
    
    # Determine output path
    output_path = args.output
    if output_path is None:
        ext = ".json" if args.format == "json" else ".txt" if args.format == "llm" else ".md"
        output_path = str(pdf_path.with_suffix(ext))
    
    # Save
    extractor.save(extracted, output_path, format=args.format)
    
    print(f"\nExtraction complete! Output saved to: {output_path}")
    print(f"  - Pages processed: {len(extracted.pages)}")
    print(f"  - Images extracted: {sum(len(p.images) for p in extracted.pages)}")
    print(f"  - Tables extracted: {sum(len(p.tables) for p in extracted.pages)}")
    
    if args.vlm:
        vlm_count = sum(
            1 for p in extracted.pages 
            for img in p.images 
            if img.vlm_description
        )
        print(f"  - VLM descriptions: {vlm_count}")
