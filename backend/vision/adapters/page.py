"""
Page Content Extraction Adapter for Follow-Up System
Extracts readable content from browser windows.
"""
import logging
from typing import Optional, Dict, Any
import asyncio

logger = logging.getLogger(__name__)


async def get_page_title(window_id: str) -> Optional[str]:
    """
    Get the title of a browser window.

    Args:
        window_id: Browser window identifier

    Returns:
        Page title, or None if unavailable
    """
    try:
        # Try AppleScript for macOS browsers
        if await _is_macos():
            return await _get_title_applescript(window_id)

        # Fallback: extract from OCR
        return await _get_title_from_ocr(window_id)

    except Exception as e:
        logger.error(f"[PAGE] Failed to get page title for {window_id}: {e}", exc_info=True)
        return None


async def get_readable_text(window_id: str, limit_chars: int = 1000) -> str:
    """
    Extract readable text content from a browser window.

    Args:
        window_id: Browser window identifier
        limit_chars: Maximum characters to return

    Returns:
        Extracted readable text
    """
    try:
        # Try OCR-based extraction
        text = await _extract_page_text_ocr(window_id)

        if text:
            # Clean up text
            text = _clean_web_text(text)

            # Limit length
            if len(text) > limit_chars:
                text = text[:limit_chars] + "..."

            return text

        logger.warning(f"[PAGE] No readable text extracted from {window_id}")
        return ""

    except Exception as e:
        logger.error(f"[PAGE] Failed to extract page text for {window_id}: {e}", exc_info=True)
        return ""


async def extract_page_content(window_id: str, snapshot_id: str) -> Dict[str, Any]:
    """
    Extract comprehensive page content.

    Args:
        window_id: Browser window identifier
        snapshot_id: Associated screenshot snapshot

    Returns:
        Dict with title, text, links, etc.
    """
    try:
        content = {
            "title": await get_page_title(window_id),
            "text": await get_readable_text(window_id, limit_chars=800),
            "links": await _extract_links(snapshot_id),
            "has_forms": await _detect_forms(snapshot_id),
            "has_errors": await _detect_error_pages(snapshot_id),
        }

        logger.info(
            f"[PAGE] Extracted content: title={bool(content['title'])}, "
            f"text_len={len(content['text'])}, "
            f"links={len(content['links'])}"
        )

        return content

    except Exception as e:
        logger.error(f"[PAGE] Failed to extract page content: {e}", exc_info=True)
        return {
            "title": None,
            "text": "",
            "links": [],
            "has_forms": False,
            "has_errors": False,
        }


# ═══════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════

async def _is_macos() -> bool:
    """Check if running on macOS."""
    import platform
    return platform.system() == "Darwin"


async def _get_title_applescript(window_id: str) -> Optional[str]:
    """Extract page title using AppleScript (macOS only)."""
    try:
        # Try Safari
        script = f"""
        tell application "Safari"
            set windowCount to count of windows
            if windowCount > 0 then
                return name of current tab of front window
            end if
        end tell
        """

        proc = await asyncio.create_subprocess_shell(
            f"osascript -e '{script}'",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, _ = await proc.communicate()

        if proc.returncode == 0 and stdout:
            title = stdout.decode('utf-8').strip()
            logger.debug(f"[PAGE] Got title via AppleScript: {title}")
            return title

    except Exception as e:
        logger.debug(f"[PAGE] AppleScript title extraction failed: {e}")

    return None


async def _get_title_from_ocr(window_id: str) -> Optional[str]:
    """Extract page title from OCR (fallback method)."""
    try:
        from backend.vision.adapters.ocr import ocr_text_from_snapshot

        # Assume window_id can be used as snapshot_id
        text = await ocr_text_from_snapshot(window_id)

        if not text:
            return None

        # Look for title-like text (first significant line)
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        for line in lines[:5]:  # Check first 5 lines
            # Skip navigation/UI text
            if len(line) > 10 and not line.startswith(('http', 'www')):
                logger.debug(f"[PAGE] Inferred title from OCR: {line[:50]}...")
                return line[:100]  # Limit title length

    except Exception as e:
        logger.debug(f"[PAGE] OCR title extraction failed: {e}")

    return None


async def _extract_page_text_ocr(window_id: str) -> str:
    """Extract page text using OCR."""
    try:
        from backend.vision.adapters.ocr import ocr_text_from_snapshot

        text = await ocr_text_from_snapshot(window_id)
        return text

    except Exception as e:
        logger.error(f"[PAGE] OCR text extraction failed: {e}")
        return ""


def _clean_web_text(text: str) -> str:
    """Clean up OCR text from web pages."""
    import re

    # Remove navigation/UI noise
    noise_patterns = [
        r'http[s]?://\S+',  # URLs
        r'www\.\S+',
        r'[\w\.-]+@[\w\.-]+',  # Emails
        r'^\s*(Home|About|Contact|Menu|Login|Sign Up)\s*$',  # Common nav items
    ]

    cleaned = text
    for pattern in noise_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE | re.IGNORECASE)

    # Remove excessive whitespace
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
    cleaned = re.sub(r'  +', ' ', cleaned)

    return cleaned.strip()


async def _extract_links(snapshot_id: str) -> list[str]:
    """Extract visible links from page (OCR-based)."""
    try:
        from backend.vision.adapters.ocr import ocr_text_from_snapshot
        import re

        text = await ocr_text_from_snapshot(snapshot_id)

        # Find URL-like patterns
        url_pattern = r'(?:http[s]?://|www\.)\S+'
        links = re.findall(url_pattern, text, re.IGNORECASE)

        # Deduplicate and limit
        unique_links = list(dict.fromkeys(links))[:10]

        logger.debug(f"[PAGE] Extracted {len(unique_links)} links")
        return unique_links

    except Exception as e:
        logger.debug(f"[PAGE] Link extraction failed: {e}")
        return []


async def _detect_forms(snapshot_id: str) -> bool:
    """Detect if page has forms (heuristic)."""
    try:
        from backend.vision.adapters.ocr import ocr_text_from_snapshot

        text = await ocr_text_from_snapshot(snapshot_id)

        form_keywords = ['submit', 'login', 'sign in', 'register', 'email', 'password']
        text_lower = text.lower()

        return any(keyword in text_lower for keyword in form_keywords)

    except Exception:
        return False


async def _detect_error_pages(snapshot_id: str) -> bool:
    """Detect if page is an error page."""
    try:
        from backend.vision.adapters.ocr import ocr_text_from_snapshot

        text = await ocr_text_from_snapshot(snapshot_id)

        error_patterns = [
            r'404',
            r'not found',
            r'error occurred',
            r'page unavailable',
            r'access denied',
            r'forbidden',
        ]

        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in error_patterns)

    except Exception:
        return False
