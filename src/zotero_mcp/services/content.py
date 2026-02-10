"""Content retrieval service — returns structured data, not markdown."""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass

from zotero_mcp.client import (
    convert_to_markdown,
    format_item_metadata,
    get_attachment_details,
)

logger = logging.getLogger(__name__)


@dataclass
class FulltextResult:
    """Structured result from fulltext retrieval."""

    metadata_md: str | None = None
    fulltext: str | None = None
    error: str | None = None


def fetch_item_fulltext(item_key: str, *, zot=None) -> FulltextResult:
    """Fetch fulltext for a Zotero item. Returns structured data.

    Args:
        item_key: Zotero item key.
        zot: Optional pre-configured Zotero client (for testing).

    Returns:
        FulltextResult with metadata, fulltext, and/or error.
    """
    if zot is None:
        from zotero_mcp.client import get_zotero_client

        zot = get_zotero_client()

    # Get item metadata
    try:
        item = zot.item(item_key)
    except Exception as e:
        return FulltextResult(error=f"Error fetching item {item_key}: {e}")

    if not item:
        return FulltextResult(error=f"No item found with key: {item_key}")

    metadata_md = format_item_metadata(item, include_abstract=True)

    # Find attachment
    attachment = get_attachment_details(zot, item)
    if not attachment:
        return FulltextResult(metadata_md=metadata_md)

    # Try Zotero fulltext index first
    try:
        full_text_data = zot.fulltext_item(attachment.key)
        if full_text_data and "content" in full_text_data and full_text_data["content"]:
            return FulltextResult(
                metadata_md=metadata_md,
                fulltext=full_text_data["content"],
            )
    except Exception as e:
        logger.debug("Couldn't retrieve indexed full text: %s", e)

    # Fallback: download and convert
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(
                tmpdir, attachment.filename or f"{attachment.key}.pdf"
            )
            zot.dump(attachment.key, filename=os.path.basename(file_path), path=tmpdir)

            if os.path.exists(file_path):
                converted = convert_to_markdown(file_path)
                return FulltextResult(metadata_md=metadata_md, fulltext=converted)
            else:
                return FulltextResult(
                    metadata_md=metadata_md,
                    error="File download failed.",
                )
    except Exception as e:
        return FulltextResult(
            metadata_md=metadata_md,
            error=f"Error accessing attachment: {e}",
        )
