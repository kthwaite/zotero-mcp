"""Tests for search_notes -- verifies it uses structured annotation data."""

from unittest.mock import MagicMock, patch


def test_search_notes_includes_matching_annotations():
    """search_notes should use structured annotations, not parse markdown."""
    from zotero_mcp.services.annotations import AnnotationsResult

    mock_zot = MagicMock()
    # Return notes matching the query
    mock_zot.items.return_value = [
        {
            "key": "NOTE1",
            "data": {
                "itemType": "note",
                "note": "<p>Discussion of neural networks in biology</p>",
                "parentItem": "PAPER1",
                "tags": [{"tag": "neuro"}],
            },
        }
    ]

    fake_annotations = AnnotationsResult(
        annotations=[
            {
                "key": "ANN1",
                "annotation_type": "highlight",
                "text": "neural networks are fundamental",
                "comment": "",
                "color": "#ff0000",
                "parent_item": "PAPER1",
                "tags": [],
                "page": 3,
                "page_label": "3",
                "attachment_title": "paper.pdf",
                "color_category": "Red",
                "source": "zotero_api",
                "has_image": False,
            },
            {
                "key": "ANN2",
                "annotation_type": "highlight",
                "text": "unrelated protein folding",
                "comment": "",
                "color": "#00ff00",
                "parent_item": "PAPER2",
                "tags": [],
                "page": 10,
                "page_label": "10",
                "attachment_title": None,
                "color_category": None,
                "source": "zotero_api",
                "has_image": False,
            },
        ],
    )

    with (
        patch("zotero_mcp.server.get_zotero_client", return_value=mock_zot),
        patch("zotero_mcp.server.fetch_annotations", return_value=fake_annotations),
    ):
        from zotero_mcp.server import search_notes

        fn = search_notes.fn if hasattr(search_notes, "fn") else search_notes
        ctx = MagicMock()
        result = fn(query="neural", ctx=ctx)

    # Should include the matching note
    assert "NOTE1" in result
    # Should include the matching annotation (contains "neural")
    assert "neural networks are fundamental" in result
    # Should NOT include the unrelated annotation
    assert "protein folding" not in result


def test_search_notes_empty_query():
    """Empty query returns error."""
    from zotero_mcp.server import search_notes

    fn = search_notes.fn if hasattr(search_notes, "fn") else search_notes
    ctx = MagicMock()
    result = fn(query="  ", ctx=ctx)
    assert "Error" in result or "empty" in result.lower()
