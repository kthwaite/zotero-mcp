"""Annotation retrieval service — returns structured data, not markdown."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TypedDict

logger = logging.getLogger(__name__)


class AnnotationRecord(TypedDict):
    """Normalized annotation record returned by fetch_annotations."""

    key: str
    annotation_type: str
    text: str
    comment: str
    color: str
    parent_item: str
    tags: list[str]
    page: int | None
    page_label: str | None
    attachment_title: str | None
    color_category: str | None
    source: str
    has_image: bool


@dataclass
class AnnotationsResult:
    """Structured result from annotation retrieval."""

    annotations: list[AnnotationRecord] = field(default_factory=list)
    parent_title: str = "Untitled Item"
    error: str | None = None


def _normalize_annotation(raw: dict) -> AnnotationRecord:
    """Normalize a Zotero annotation dict into a flat record."""
    data = raw.get("data", {})
    return {
        "key": raw.get("key", ""),
        "annotation_type": data.get("annotationType", "unknown"),
        "text": data.get("annotationText", ""),
        "comment": data.get("annotationComment", ""),
        "color": data.get("annotationColor", ""),
        "parent_item": data.get("parentItem", ""),
        "tags": [t.get("tag", "") for t in (data.get("tags") or [])],
        "page": data.get("_pdf_page"),
        "page_label": data.get("_pageLabel"),
        "attachment_title": data.get("_attachment_title"),
        "color_category": data.get("_color_category"),
        "source": (
            "better_bibtex"
            if data.get("_from_better_bibtex")
            else "pdf_extraction"
            if data.get("_from_pdf_extraction")
            else "zotero_api"
        ),
        "has_image": bool(data.get("_image_path")),
    }


def _fetch_bbt_annotations(parent: dict, item_key: str) -> list[dict]:
    """Try to fetch annotations via Better BibTeX. Returns raw Zotero-shaped dicts."""
    try:
        from zotero_mcp.better_bibtex_client import (
            ZoteroBetterBibTexAPI,
            get_color_category,
            process_annotation,
        )
    except ImportError:
        return []

    try:
        bibtex = ZoteroBetterBibTexAPI()
    except Exception:
        return []

    if not bibtex.is_zotero_running():
        return []

    # Extract citation key
    citation_key = None
    extra_field = parent["data"].get("extra", "")
    for line in extra_field.split("\n"):
        low = line.lower()
        if low.startswith("citation key:"):
            citation_key = line.split(":", 1)[1].strip()
            break
        elif low.startswith("citationkey:"):
            citation_key = line.split(":", 1)[1].strip()
            break

    if not citation_key:
        title = parent["data"].get("title", "")
        if title:
            try:
                results = bibtex.search_citekeys(title)
                for r in results:
                    if r.get("citekey"):
                        citation_key = r["citekey"]
                        break
            except Exception:
                pass

    if not citation_key:
        return []

    results = []
    try:
        search_results = bibtex._make_request("item.search", [citation_key])
        library = "*"
        if search_results:
            matched = next(
                (i for i in search_results if i.get("citekey") == citation_key),
                None,
            )
            if matched:
                library = matched.get("library", "*")

        attachments = bibtex.get_attachments(citation_key, library)

        for attachment in attachments:
            raw_annotations = bibtex.get_annotations_from_attachment(attachment)
            for anno in raw_annotations:
                processed = process_annotation(anno, attachment)
                if processed:
                    results.append(
                        {
                            "key": processed.get("id", ""),
                            "data": {
                                "itemType": "annotation",
                                "annotationType": processed.get("type", "highlight"),
                                "annotationText": processed.get("annotatedText", ""),
                                "annotationComment": processed.get("comment", ""),
                                "annotationColor": processed.get("color", ""),
                                "parentItem": item_key,
                                "tags": [],
                                "_pdf_page": processed.get("page", 0),
                                "_pageLabel": processed.get("pageLabel", ""),
                                "_attachment_title": attachment.get("title", ""),
                                "_color_category": get_color_category(
                                    processed.get("color", "")
                                ),
                                "_from_better_bibtex": True,
                            },
                        }
                    )

        logger.debug("Retrieved %d annotations via Better BibTeX", len(results))
    except Exception as e:
        logger.debug("Error fetching BBT annotations: %s", e)

    return results


def _fetch_pdf_annotations(zot, item_key: str) -> list[dict]:
    """Try to extract annotations directly from PDF attachments."""
    import tempfile
    import uuid

    try:
        from zotero_mcp.pdfannots_helper import (
            ensure_pdfannots_installed,
            extract_annotations_from_pdf,
        )
    except ImportError:
        return []

    if not ensure_pdfannots_installed():
        return []

    results = []
    try:
        children = zot.children(item_key)
        pdf_attachments = [
            c
            for c in children
            if c.get("data", {}).get("contentType") == "application/pdf"
        ]

        for attachment in pdf_attachments:
            with tempfile.TemporaryDirectory() as tmpdir:
                att_key = attachment.get("key", "")
                file_path = os.path.join(tmpdir, f"{att_key}.pdf")
                zot.dump(att_key, file_path)

                if os.path.exists(file_path):
                    extracted = extract_annotations_from_pdf(file_path, tmpdir)
                    for ext in extracted:
                        if not ext.get("annotatedText") and not ext.get("comment"):
                            continue
                        results.append(
                            {
                                "key": f"pdf_{att_key}_{ext.get('id', uuid.uuid4().hex[:8])}",
                                "data": {
                                    "itemType": "annotation",
                                    "annotationType": ext.get("type", "highlight"),
                                    "annotationText": ext.get("annotatedText", ""),
                                    "annotationComment": ext.get("comment", ""),
                                    "annotationColor": ext.get("color", ""),
                                    "parentItem": item_key,
                                    "tags": [],
                                    "_pdf_page": ext.get("page", 0),
                                    "_from_pdf_extraction": True,
                                    "_attachment_title": attachment.get("data", {}).get(
                                        "title", "PDF"
                                    ),
                                },
                            }
                        )
    except Exception as e:
        logger.debug("Error extracting PDF annotations: %s", e)

    return results


def fetch_annotations(
    item_key: str | None = None,
    use_pdf_extraction: bool = False,
    limit: int | None = None,
    *,
    zot=None,
) -> AnnotationsResult:
    """Fetch annotations, returning structured data.

    Args:
        item_key: Optional item key to scope annotations.
        use_pdf_extraction: Whether to try PDF extraction as fallback.
        limit: Max annotations (for library-wide queries).
        zot: Optional pre-configured Zotero client.

    Returns:
        AnnotationsResult with list of normalized annotation dicts.
    """
    if zot is None:
        from zotero_mcp.client import get_zotero_client

        zot = get_zotero_client()

    if item_key:
        # Verify item exists
        try:
            parent = zot.item(item_key)
            parent_title = parent["data"].get("title", "Untitled Item")
        except Exception as e:
            return AnnotationsResult(error=f"No item found with key {item_key}: {e}")

        raw_annotations = []

        # Try BBT first (local only)
        if os.environ.get("ZOTERO_LOCAL", "").lower() in ("true", "yes", "1"):
            raw_annotations = _fetch_bbt_annotations(parent, item_key)

        # Fallback: Zotero API
        if not raw_annotations:
            try:
                children = zot.children(item_key)
                raw_annotations = [
                    c
                    for c in children
                    if c.get("data", {}).get("itemType") == "annotation"
                ]
            except Exception as e:
                logger.debug("Error fetching API annotations: %s", e)

        # Fallback: PDF extraction
        if use_pdf_extraction and not raw_annotations:
            raw_annotations = _fetch_pdf_annotations(zot, item_key)

        normalized = [_normalize_annotation(a) for a in raw_annotations]
        return AnnotationsResult(annotations=normalized, parent_title=parent_title)

    else:
        # Library-wide query
        zot.add_parameters(itemType="annotation", limit=limit or 50)
        raw_annotations = zot.everything(zot.items())
        normalized = [_normalize_annotation(a) for a in raw_annotations]
        return AnnotationsResult(annotations=normalized)
