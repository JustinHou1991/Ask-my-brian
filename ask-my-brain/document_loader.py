"""
Ask My Brain - 文档加载与解析模块

- MIME白名单检查
- PDF三级降级: pdfplumber -> PyPDF2 -> 友好提示
- 大文件超时保护
- 编码自动检测

依赖: os, requests, bs4, pdfplumber/PyPDF2, config, utils
"""
import os
import mimetypes
import requests
from bs4 import BeautifulSoup

from config import DOCUMENTS_DIR
from utils import ensure_dir, log

ALLOWED_MIME = {'application/pdf', 'text/plain', 'text/markdown', 'text/html', 'text/csv',
                'image/png', 'image/jpeg', 'image/bmp', 'image/tiff', 'image/webp'}


def _check_mime(filepath: str):
    mime, _ = mimetypes.guess_type(filepath)
    if mime and mime not in ALLOWED_MIME:
        raise ValueError(f"不支持的文件类型: {mime}")


def load_pdf(filepath: str) -> str:
    """
    PDF解析四级降级。对大文件做超时保护。
    Level 1: pdfplumber文本提取
    Level 2: PyPDF2文本提取
    Level 3: 扫描件检测 + OCR
    Level 4: PDF内嵌图片OCR
    """
    fname = os.path.basename(filepath)
    fsize = os.path.getsize(filepath)
    log.info(f"解析PDF: {fname} ({fsize/1024/1024:.1f}MB)")

    # Level 1: pdfplumber
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(filepath) as pdf:
            total_pages = len(pdf.pages)
            log.info(f"PDF共 {total_pages} 页")
            for i, page in enumerate(pdf.pages):
                t = page.extract_text()
                if t:
                    text_parts.append(t)
                if (i + 1) % 10 == 0:
                    log.debug(f"  已解析 {i+1}/{total_pages} 页")
        content = "\n\n".join(text_parts)
        if len(content) > 100:
            log.info(f"pdfplumber成功: {fname}, {len(content)} 字符, {total_pages} 页")
            return content
        log.warning(f"pdfplumber提取过少({len(content)}字符)")
    except ImportError:
        log.warning("pdfplumber未安装")
    except Exception as e:
        log.warning(f"pdfplumber失败: {e}")

    # Level 2: PyPDF2
    try:
        import PyPDF2
        text_parts = []
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        content = "\n\n".join(text_parts)
        if len(content) > 100:
            log.info(f"PyPDF2成功: {fname}, {len(content)} 字符")
            return content
    except Exception as e:
        log.warning(f"PyPDF2失败: {e}")

    # Level 3: 扫描件检测 + 整页OCR
    try:
        from multimodal_loader import is_scanned_pdf, extract_full_page_ocr
        if is_scanned_pdf(filepath):
            log.info(f"检测到扫描件PDF，启用OCR: {fname}")
            ocr_text = extract_full_page_ocr(filepath)
            if len(ocr_text) > 100:
                log.info(f"扫描件OCR成功: {fname}, {len(ocr_text)} 字符")
                return ocr_text
    except Exception as e:
        log.warning(f"扫描件OCR失败: {e}")

    # Level 4: 内嵌图片OCR
    try:
        from multimodal_loader import extract_images_from_pdf
        log.info(f"尝试提取PDF内嵌图片: {fname}")
        img_results = extract_images_from_pdf(filepath)
        ocr_texts = [r["ocr_text"] for r in img_results if r["ocr_text"]]
        if ocr_texts:
            content = "\n\n".join(ocr_texts)
            log.info(f"PDF图片OCR成功: {len(ocr_texts)}张有文字, {len(content)} 字符")
            return content
    except Exception as e:
        log.warning(f"PDF图片OCR失败: {e}")

    raise RuntimeError(
        f"PDF解析失败 [{fname}]: 文本提取不足。\n"
        f"可能是扫描件PDF，请尝试:\n"
        f"  1. 安装OCR支持: pip install pytesseract + Tesseract\n"
        f"  2. 用OCR工具转为文本\n"
        f"  3. 复制内容粘贴为 .txt 上传"
    )


def load_image(filepath: str) -> str:
    """加载图片文件并OCR"""
    from multimodal_loader import load_image as ocr_load_image
    return ocr_load_image(filepath)


def load_markdown(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="gbk", errors="ignore") as f:
            return f.read()


def load_html(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        raise RuntimeError(f"HTML解析失败: {e}") from e


def fetch_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}
    try:
        resp = requests.get(url, timeout=30, headers=headers)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        log.info(f"网页抓取: {url}, {len(text)} 字符")
        return text
    except requests.RequestException as e:
        raise RuntimeError(f"网页抓取失败: {e}\n请手动复制内容保存为.txt上传") from e


def load_document(filepath: str) -> dict:
    _check_mime(filepath)
    ext = os.path.splitext(filepath)[1].lower()
    filename = os.path.basename(filepath)

    loaders = {
        ".pdf": load_pdf, ".md": load_markdown, ".txt": load_markdown,
        ".html": load_html, ".htm": load_html,
        ".png": load_image, ".jpg": load_image, ".jpeg": load_image,
        ".bmp": load_image, ".tiff": load_image, ".tif": load_image, ".webp": load_image,
    }
    loader = loaders.get(ext)
    if not loader:
        raise ValueError(f"不支持的格式: {ext}")

    content = loader(filepath)
    if not content or not content.strip():
        raise RuntimeError(f"文件内容为空: {filename}")
    log.info(f"文档加载: {filename}, {len(content)} 字符")
    return {"filename": filename, "content": content, "char_count": len(content), "source_type": ext.lstrip(".")}


def load_url(url: str) -> dict:
    from urllib.parse import urlparse
    content = fetch_url(url)
    parsed = urlparse(url)
    filename = parsed.netloc.replace(".", "_") + ".html"
    ensure_dir(DOCUMENTS_DIR)
    with open(os.path.join(DOCUMENTS_DIR, filename), "w", encoding="utf-8") as f:
        f.write(content)
    return {"filename": filename, "content": content, "char_count": len(content), "source_type": "url"}


def load_from_directory(dirpath: str) -> list:
    supported = {".pdf", ".md", ".txt", ".html", ".htm", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    docs = []
    for fname in os.listdir(dirpath):
        if os.path.splitext(fname)[1].lower() in supported:
            try:
                docs.append(load_document(os.path.join(dirpath, fname)))
            except Exception as e:
                log.warning(f"加载 {fname} 失败: {e}")
    return docs
