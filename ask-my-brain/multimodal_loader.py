"""
Ask My Brain - 多模态文档加载模块

支持图片OCR解析和扫描件PDF处理。
- PDF内嵌图片提取与OCR
- 独立图片文件OCR（PNG/JPG/BMP/TIFF）
- 图片预处理（灰度、放大、二值化）提升识别率
- OCR引擎可选：pytesseract（默认）/ PaddleOCR

依赖: PIL, numpy, pdfplumber, utils, config
"""
import os
import io
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from utils import log, ensure_dir

# OCR引擎选择（优先PaddleOCR，降级到pytesseract）
_ocr_engine = None


def _detect_ocr_engine():
    """检测可用的OCR引擎"""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return "tesseract"
    except Exception:
        pass
    try:
        import paddleocr
        return "paddleocr"
    except ImportError:
        pass
    return None


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = _detect_ocr_engine()
        if _ocr_engine:
            log.info(f"OCR引擎: {_ocr_engine}")
        else:
            log.warning("未检测到OCR引擎（pytesseract/PaddleOCR），图片文字提取将不可用")
    return _ocr_engine


def preprocess_image(img: Image.Image) -> Image.Image:
    """图片预处理，提升OCR识别率"""
    w, h = img.size
    if w < 300 or h < 300:
        scale = max(300 / w, 300 / h, 1.0)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    if img.mode != "L":
        img = img.convert("L")
    img = img.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    img = img.point(lambda x: 255 if x > 128 else 0)
    return img


def ocr_image(img: Image.Image, lang: str = "chi_sim+eng") -> str:
    """对单张图片执行OCR文字识别"""
    engine = _get_ocr_engine()
    if engine is None:
        return ""
    processed = preprocess_image(img)
    try:
        if engine == "tesseract":
            import pytesseract
            return pytesseract.image_to_string(processed, lang=lang)
        elif engine == "paddleocr":
            import paddleocr
            if not hasattr(ocr_image, "_paddle"):
                ocr_image._paddle = paddleocr.PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            img_array = np.array(processed)
            result = ocr_image._paddle.ocr(img_array, cls=True)
            if result and result[0]:
                return "\n".join([line[1][0] for line in result[0]])
    except Exception as e:
        log.warning(f"OCR识别失败: {e}")
    return ""


def extract_images_from_pdf(pdf_path: str, save_dir: str = None) -> list:
    """
    从PDF中提取内嵌图片并执行OCR。

    Args:
        pdf_path: PDF文件路径
        save_dir: 图片保存目录（可选）

    Returns:
        list: [{"image_path": str, "page": int, "ocr_text": str}]
    """
    import pdfplumber
    if save_dir:
        ensure_dir(save_dir)

    results = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                for img_idx, img_info in enumerate(page.images):
                    try:
                        x0 = max(0, img_info["x0"])
                        y0 = max(0, img_info["top"])
                        x1 = min(page.width, img_info["x1"])
                        y1 = min(page.height, img_info["bottom"])
                        if x1 - x0 < 10 or y1 - y0 < 10:
                            continue
                        cropped = page.crop((x0, y0, x1, y1))
                        pil_img = cropped.to_image(resolution=150).original
                        ocr_text = ocr_image(pil_img)
                        img_path = None
                        if save_dir:
                            img_path = os.path.join(save_dir, f"page{page_num+1}_img{img_idx+1}.png")
                            pil_img.save(img_path)
                        results.append({
                            "image_path": img_path,
                            "page": page_num + 1,
                            "ocr_text": ocr_text,
                        })
                    except Exception as e:
                        log.debug(f"页面{page_num+1}图片{img_idx+1}提取失败: {e}")
    except Exception as e:
        log.error(f"PDF图片提取失败: {e}")

    log.info(f"PDF图片OCR: {len(results)}张图片, {sum(1 for r in results if r['ocr_text'])}张有文字")
    return results


def extract_full_page_ocr(pdf_path: str, page_numbers: list = None) -> str:
    """
    对PDF整页进行OCR（适用于扫描件PDF）。

    Args:
        pdf_path: PDF文件路径
        page_numbers: 需要OCR的页码列表（None=全部）

    Returns:
        str: OCR提取的文字
    """
    import pdfplumber
    all_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            pages_to_process = page_numbers if page_numbers else range(total_pages)
            for page_num in pages_to_process:
                if page_num >= total_pages:
                    continue
                page = pdf.pages[page_num]
                pil_img = page.to_image(resolution=200).original
                text = ocr_image(pil_img)
                if text.strip():
                    all_text.append(f"[第{page_num+1}页 OCR]\n{text}")
                if (page_num + 1) % 10 == 0:
                    log.debug(f"OCR进度: {page_num+1}/{total_pages}")
    except Exception as e:
        log.error(f"整页OCR失败: {e}")

    result = "\n\n".join(all_text)
    log.info(f"整页OCR完成: {len(all_text)}页有文字, 共{len(result)}字符")
    return result


def load_image(filepath: str) -> str:
    """加载独立图片文件并OCR"""
    try:
        img = Image.open(filepath)
        text = ocr_image(img)
        log.info(f"图片OCR: {os.path.basename(filepath)}, {len(text)}字符")
        return f"[图片: {os.path.basename(filepath)}]\n{text}" if text else ""
    except Exception as e:
        raise RuntimeError(f"图片加载失败: {e}") from e


def is_scanned_pdf(pdf_path: str, sample_pages: int = 3) -> bool:
    """
    判断PDF是否为扫描件（文本极少但有图片）。
    """
    import pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_text = 0
            total_images = 0
            for i, page in enumerate(pdf.pages[:sample_pages]):
                text = page.extract_text() or ""
                total_text += len(text.strip())
                total_images += len(page.images)
            avg_text = total_text / min(sample_pages, len(pdf.pages))
            return avg_text < 50 and total_images > 0
    except Exception:
        return False
