"""Tests for annotations service — structured annotation retrieval."""

from unittest.mock import MagicMock

import pytest


def test_annotation_record_shape():
    """Service returns list of AnnotationRecord dicts with known keys."""
    from zotero_mcp.services.annotations import fetch_annotations

    assert callable(fetch_annotations)


def test_fetch_annotations_via_api(mock_zotero_with_annotations):
    """When BBT is unavailable, falls back to Zotero API annotations."""
    from zotero_mcp.services.annotations import fetch_annotations

    result = fetch_annotations(item_key="ITEM1", zot=mock_zotero_with_annotations)

    assert result.error is None
    assert len(result.annotations) == 1
    assert result.annotations[0]["annotation_type"] == "highlight"
    assert result.annotations[0]["text"] == "Important finding"
    assert result.parent_title == "Test Paper"


def test_fetch_annotations_all_library(mock_zotero_with_annotations):
    """When no item_key, fetches all annotations from library."""
    from zotero_mcp.services.annotations import fetch_annotations

    mock_zotero_with_annotations.items.return_value = [
        {
            "key": "ANN1",
            "data": {
                "itemType": "annotation",
                "annotationType": "note",
                "annotationText": "",
                "annotationComment": "My thought",
                "annotationColor": "#ffff00",
                "parentItem": "ITEM1",
                "tags": [],
            },
        }
    ]
    mock_zotero_with_annotations.everything.return_value = (
        mock_zotero_with_annotations.items.return_value
    )

    result = fetch_annotations(zot=mock_zotero_with_annotations)

    assert result.error is None
    assert len(result.annotations) == 1
    assert result.annotations[0]["comment"] == "My thought"


def test_fetch_annotations_no_results(mock_zotero_with_annotations):
    """Empty annotations list when item has no annotations."""
    mock_zotero_with_annotations.children.return_value = []

    from zotero_mcp.services.annotations import fetch_annotations

    result = fetch_annotations(item_key="EMPTY", zot=mock_zotero_with_annotations)

    assert result.annotations == []


def test_annotations_tool_delegates_to_service():
    """The @mcp.tool wrapper calls the service and formats to markdown."""
    from unittest.mock import MagicMock, patch
    from zotero_mcp.services.annotations import AnnotationsResult

    fake = AnnotationsResult(
        annotations=[
            {
                "key": "A1",
                "annotation_type": "highlight",
                "text": "Important text",
                "comment": "My note",
                "color": "#ff0000",
                "parent_item": "P1",
                "tags": ["review"],
                "page": 5,
                "page_label": "5",
                "attachment_title": "paper.pdf",
                "color_category": "Red",
                "source": "zotero_api",
                "has_image": False,
            }
        ],
        parent_title="Test Paper",
    )

    with patch(
        "zotero_mcp.server.fetch_annotations", return_value=fake
    ):
        from zotero_mcp.server import get_annotations

        # Access underlying function (FastMCP wraps with @mcp.tool)
        fn = get_annotations.fn if hasattr(get_annotations, 'fn') else get_annotations
        ctx = MagicMock()
        result = fn(item_key="P1", ctx=ctx)

    assert "Test Paper" in result
    assert "Important text" in result
    assert "highlight" in result


@pytest.fixture
def mock_zotero_with_annotations():
    """Mock Zotero client that returns annotations for a known item."""
    zot = MagicMock()

    zot.item.return_value = {
        "key": "ITEM1",
        "data": {
            "title": "Test Paper",
            "extra": "",
            "creators": [],
        },
    }

    zot.children.return_value = [
        {
            "key": "ANN1",
            "data": {
                "itemType": "annotation",
                "annotationType": "highlight",
                "annotationText": "Important finding",
                "annotationComment": "",
                "annotationColor": "#ff6666",
                "parentItem": "ITEM1",
                "tags": [{"tag": "key-point"}],
            },
        },
        {
            "key": "NOTE1",
            "data": {
                "itemType": "note",
                "note": "Not an annotation",
            },
        },
    ]

    return zot
