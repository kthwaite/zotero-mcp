"""Tests for ChatGPT connector wrappers."""

import json
from unittest.mock import MagicMock, patch


def test_connector_fetch_uses_service():
    """connector_fetch should call fetch_item_fulltext service, not the tool."""
    from zotero_mcp.services.content import FulltextResult

    fake_result = FulltextResult(
        metadata_md="# Metadata",
        fulltext="The actual full text of the paper with enough content to exceed the minimum length threshold used by the connector.",
    )

    mock_zot = MagicMock()
    mock_zot.item.return_value = {
        "key": "XYZ789",
        "data": {
            "itemType": "journalArticle",
            "title": "Test Paper",
            "date": "2025",
            "DOI": "10.1234/test",
            "creators": [{"firstName": "A", "lastName": "B", "creatorType": "author"}],
            "tags": [{"tag": "ml"}],
        },
    }

    with (
        patch("zotero_mcp.server.fetch_item_fulltext", return_value=fake_result),
        patch("zotero_mcp.server.get_zotero_client", return_value=mock_zot),
    ):
        from zotero_mcp.server import connector_fetch

        # Access the underlying function (FastMCP wraps it)
        fn = connector_fetch.fn if hasattr(connector_fetch, 'fn') else connector_fetch
        ctx = MagicMock()
        raw = fn(id="XYZ789", ctx=ctx)

    data = json.loads(raw)
    assert data["id"] == "XYZ789"
    assert data["title"] == "Test Paper"
    # Should use the fulltext from the service, not parse markdown
    assert "The actual full text" in data["text"]
