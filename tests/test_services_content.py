"""Tests for content service — fulltext retrieval logic."""

from unittest.mock import MagicMock, patch

import pytest


def test_fetch_fulltext_returns_dataclass():
    """Service returns a FulltextResult, not a markdown string."""
    from zotero_mcp.services.content import FulltextResult, fetch_item_fulltext

    assert hasattr(FulltextResult, "metadata_md")
    assert hasattr(FulltextResult, "fulltext")
    assert hasattr(FulltextResult, "error")
    assert callable(fetch_item_fulltext)


def test_fetch_fulltext_from_index(mock_zotero_client):
    """When Zotero's fulltext index has content, return it."""
    from zotero_mcp.services.content import fetch_item_fulltext

    result = fetch_item_fulltext("ABC123", zot=mock_zotero_client)

    assert result.fulltext == "Indexed fulltext content."
    assert result.error is None
    assert "Test Article" in result.metadata_md


def test_fetch_fulltext_no_item(mock_zotero_client):
    """When item doesn't exist, return error."""
    from zotero_mcp.services.content import fetch_item_fulltext

    mock_zotero_client.item.side_effect = Exception("not found")
    result = fetch_item_fulltext("MISSING", zot=mock_zotero_client)

    assert result.error is not None
    assert result.fulltext is None


def test_fetch_fulltext_no_attachment(mock_zotero_client):
    """When item has no attachment, fulltext is None but metadata exists."""
    from zotero_mcp.services.content import fetch_item_fulltext

    with patch("zotero_mcp.services.content.get_attachment_details", return_value=None):
        result = fetch_item_fulltext("ABC123", zot=mock_zotero_client)

    assert result.fulltext is None
    assert result.metadata_md is not None
    assert result.error is None


@pytest.fixture
def mock_zotero_client():
    """Mock pyzotero client with a single item that has indexed fulltext."""
    zot = MagicMock()
    zot.item.return_value = {
        "key": "ABC123",
        "data": {
            "itemType": "journalArticle",
            "title": "Test Article",
            "creators": [
                {"firstName": "A", "lastName": "Author", "creatorType": "author"}
            ],
            "date": "2025",
            "tags": [],
        },
    }

    # Mock attachment details — must return an object with .key, .content_type, .filename
    mock_attachment = MagicMock()
    mock_attachment.key = "ATT001"
    mock_attachment.content_type = "application/pdf"
    mock_attachment.filename = "article.pdf"

    # Default: attachment exists and fulltext index works
    zot.fulltext_item.return_value = {"content": "Indexed fulltext content."}

    # Make get_attachment_details return the mock by default
    with patch(
        "zotero_mcp.services.content.get_attachment_details",
        return_value=mock_attachment,
    ):
        yield zot
