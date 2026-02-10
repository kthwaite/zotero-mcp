"""
ChromaDB client for semantic search functionality.

This module provides persistent vector database storage and embedding functions
for semantic search over Zotero libraries.
"""

import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any
import logging

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings

logger = logging.getLogger(__name__)


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout temporarily."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class OpenAIEmbeddingFunction(EmbeddingFunction):
    """Custom OpenAI embedding function for ChromaDB."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        try:
            import openai

            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self.client = openai.OpenAI(**client_kwargs)
        except ImportError:
            raise ImportError("openai package is required for OpenAI embeddings")

    def name(self) -> str:
        """Return the name of this embedding function."""
        return "openai"

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using OpenAI API."""
        response = self.client.embeddings.create(model=self.model_name, input=input)
        return [data.embedding for data in response.data]


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom Gemini embedding function for ChromaDB using google-genai."""

    def __init__(
        self,
        model_name: str = "models/text-embedding-004",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model_name = model_name
        self.api_key = (
            api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        self.base_url = base_url or os.getenv("GEMINI_BASE_URL")
        if not self.api_key:
            raise ValueError("Gemini API key is required")

        try:
            from google import genai
            from google.genai import types

            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                http_options = types.HttpOptions(baseUrl=self.base_url)
                client_kwargs["http_options"] = http_options
            self.client = genai.Client(**client_kwargs)
            self.types = types
        except ImportError:
            raise ImportError("google-genai package is required for Gemini embeddings")

    def name(self) -> str:
        """Return the name of this embedding function."""
        return "gemini"

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using Gemini API."""
        embeddings = []
        for text in input:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=[text],
                config=self.types.EmbedContentConfig(
                    task_type="retrieval_document", title="Zotero library document"
                ),
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings


class HuggingFaceEmbeddingFunction(EmbeddingFunction):
    """Custom HuggingFace embedding function for ChromaDB using sentence-transformers."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        self.model_name = model_name

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name, trust_remote_code=True)
        except ImportError:
            raise ImportError(
                "sentence-transformers package is required for HuggingFace embeddings. Install with: pip install sentence-transformers"
            )

    def name(self) -> str:
        """Return the name of this embedding function."""
        return f"huggingface-{self.model_name}"

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using HuggingFace model."""
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()


class ChromaClient:
    """ChromaDB client for Zotero semantic search."""

    def __init__(
        self,
        collection_name: str = "zotero_library",
        persist_directory: str | None = None,
        embedding_model: str = "default",
        embedding_config: dict[str, Any] | None = None,
        embedding_mismatch_behavior: str = "reset",
    ):
        """
        Initialize ChromaDB client.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Model to use for embeddings ('default', 'openai', 'gemini', 'qwen', 'embeddinggemma', or HuggingFace model name)
            embedding_config: Configuration for the embedding model
            embedding_mismatch_behavior: Behavior when collection embedding conflicts with configured embedding ('reset', 'reuse', 'error')
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_config = embedding_config or {}
        self.embedding_mismatch_behavior = self._normalize_mismatch_behavior(
            embedding_mismatch_behavior
        )
        self.embedding_details = self._resolve_embedding_details()
        self.embedding_signature = self._build_embedding_signature()
        self.collection_metadata = self._build_collection_metadata()

        # Set up persistent directory
        if persist_directory is None:
            # Use user's config directory by default
            config_dir = Path.home() / ".config" / "zotero-mcp"
            config_dir.mkdir(parents=True, exist_ok=True)
            persist_directory = str(config_dir / "chroma_db")

        self.persist_directory = persist_directory

        # Initialize ChromaDB client with stdout suppression
        with suppress_stdout():
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )

            # Set up embedding function
            self.embedding_function = self._create_embedding_function()

            # Load or create collection
            self.collection = self._get_or_create_collection()

    @staticmethod
    def _normalize_mismatch_behavior(behavior: str) -> str:
        """Normalize embedding mismatch behavior with safe default."""
        normalized = str(behavior or "reset").strip().lower()
        if normalized in {"reset", "reuse", "error"}:
            return normalized

        logger.warning(
            "Unknown embedding mismatch behavior '%s'. Falling back to 'reset'.",
            behavior,
        )
        return "reset"

    def _resolve_embedding_details(self) -> dict[str, str]:
        """Resolve provider/model details for the configured embedding backend."""
        if self.embedding_model == "openai":
            return {
                "provider": "openai",
                "model_name": str(
                    self.embedding_config.get("model_name", "text-embedding-3-small")
                ),
                "base_url": str(self.embedding_config.get("base_url", "") or ""),
            }

        if self.embedding_model == "gemini":
            return {
                "provider": "gemini",
                "model_name": str(
                    self.embedding_config.get("model_name", "models/text-embedding-004")
                ),
                "base_url": str(self.embedding_config.get("base_url", "") or ""),
            }

        if self.embedding_model == "qwen":
            return {
                "provider": "huggingface",
                "model_name": str(
                    self.embedding_config.get("model_name", "Qwen/Qwen3-Embedding-0.6B")
                ),
            }

        if self.embedding_model == "embeddinggemma":
            return {
                "provider": "huggingface",
                "model_name": str(
                    self.embedding_config.get(
                        "model_name", "google/embeddinggemma-300m"
                    )
                ),
            }

        if self.embedding_model not in ["default", "openai", "gemini"]:
            # Treat any other value as a HuggingFace model name
            return {
                "provider": "huggingface",
                "model_name": str(self.embedding_model),
            }

        return {
            "provider": "chromadb-default",
            "model_name": "all-MiniLM-L6-v2",
        }

    def _build_embedding_signature(self) -> str:
        """Build a deterministic signature for collection embedding compatibility."""
        signature_payload = {
            "provider": self.embedding_details["provider"],
            "model_name": self.embedding_details["model_name"],
            "base_url": self.embedding_details.get("base_url", ""),
        }
        payload_json = json.dumps(signature_payload, sort_keys=True)
        return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

    def _build_collection_metadata(self) -> dict[str, str]:
        """Build collection metadata that records embedding identity."""
        metadata = {
            "zotero_embedding_signature": self.embedding_signature,
            "zotero_embedding_provider": self.embedding_details["provider"],
            "zotero_embedding_model_name": self.embedding_details["model_name"],
            "zotero_embedding_model": self.embedding_model,
        }

        if base_url := self.embedding_details.get("base_url"):
            metadata["zotero_embedding_base_url"] = base_url

        return metadata

    def _get_collection(self) -> Any:
        """Get existing collection using the currently configured embedding function."""
        try:
            return self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )
        except TypeError:
            # Older Chroma versions may not accept embedding_function here.
            return self.client.get_collection(name=self.collection_name)

    def _create_collection(self) -> Any:
        """Create collection and persist embedding metadata."""
        try:
            return self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata=self.collection_metadata,
            )
        except TypeError:
            # Older Chroma versions may not accept metadata on create.
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
            )
            self._set_collection_metadata(collection)
            return collection

    def _set_collection_metadata(self, collection: Any) -> None:
        """Set/update embedding metadata on an existing collection."""
        existing_metadata = getattr(collection, "metadata", None)
        if not isinstance(existing_metadata, dict):
            existing_metadata = {}

        updated_metadata = dict(existing_metadata)
        changed = False
        for key, value in self.collection_metadata.items():
            if updated_metadata.get(key) != value:
                updated_metadata[key] = value
                changed = True

        if not changed:
            return

        try:
            collection.modify(metadata=updated_metadata)
        except Exception as e:
            logger.debug(
                "Unable to update collection metadata for '%s': %s",
                self.collection_name,
                e,
            )

    def _get_collection_embedding_signature(self, collection: Any) -> str | None:
        """Read embedding signature metadata from a collection if available."""
        metadata = getattr(collection, "metadata", None)
        if not isinstance(metadata, dict):
            return None

        signature = metadata.get("zotero_embedding_signature")
        if isinstance(signature, str) and signature:
            return signature

        return None

    def _handle_embedding_mismatch(self, existing_signature: str | None) -> None:
        """Handle collection embedding mismatch according to configured behavior."""
        configured = f"{self.embedding_details['provider']}:{self.embedding_details['model_name']}"
        message = (
            f"Embedding mismatch for Chroma collection '{self.collection_name}'. "
            f"Existing signature: {existing_signature or 'unknown'}, "
            f"configured embedding: {configured}."
        )

        if self.embedding_mismatch_behavior == "reuse":
            logger.warning(
                "%s Reusing existing collection as-is; this may fail if dimensions differ.",
                message,
            )
            return

        if self.embedding_mismatch_behavior == "error":
            raise RuntimeError(
                f"{message} Run 'zotero-mcp update-db --force-rebuild' or set "
                "ZOTERO_EMBEDDING_MISMATCH_BEHAVIOR=reset."
            )

        logger.warning(
            "%s Resetting collection to match configured embedding.", message
        )
        self.reset_collection()

    @staticmethod
    def _is_dimension_mismatch_error(error: Exception) -> bool:
        """Best-effort detector for Chroma embedding dimensionality mismatch errors."""
        message = str(error).lower()
        if "dimension" not in message and "dimensionality" not in message:
            return False

        return any(token in message for token in ("embedding", "vector", "collection"))

    def _handle_runtime_embedding_mismatch(
        self, error: Exception, operation: str
    ) -> bool:
        """Handle runtime dimensionality mismatch by resetting collection (default)."""
        if not self._is_dimension_mismatch_error(error):
            return False

        if self.embedding_mismatch_behavior != "reset":
            return False

        logger.warning(
            "Detected embedding dimensionality clash during '%s' for collection '%s'. "
            "Default behavior is to reset collection and continue.",
            operation,
            self.collection_name,
        )
        self.reset_collection()
        return True

    def _get_or_create_collection(self) -> Any:
        """Get existing collection or create one with the configured embedding."""
        try:
            collection = self._get_collection()
        except Exception:
            return self._create_collection()

        existing_signature = self._get_collection_embedding_signature(collection)
        if existing_signature and existing_signature != self.embedding_signature:
            # Keep a reference so reset_collection can replace it in-place.
            self.collection = collection
            self._handle_embedding_mismatch(existing_signature)
            return self.collection

        if existing_signature:
            self._set_collection_metadata(collection)
            return collection

        # Legacy collection without embedding metadata.
        try:
            count = collection.count()
        except Exception:
            count = None

        if count == 0:
            # Safe to stamp metadata when collection is empty.
            self._set_collection_metadata(collection)
        else:
            logger.info(
                "Collection '%s' has no embedding signature metadata (legacy DB). "
                "Will keep existing data and auto-reset if dimensionality conflicts occur.",
                self.collection_name,
            )

        return collection

    def _create_embedding_function(self) -> EmbeddingFunction:
        """Create the appropriate embedding function based on configuration."""
        if self.embedding_model == "openai":
            api_key = self.embedding_config.get("api_key")
            base_url = self.embedding_details.get("base_url") or None
            return OpenAIEmbeddingFunction(
                model_name=self.embedding_details["model_name"],
                api_key=api_key,
                base_url=base_url,
            )

        if self.embedding_model == "gemini":
            api_key = self.embedding_config.get("api_key")
            base_url = self.embedding_details.get("base_url") or None
            return GeminiEmbeddingFunction(
                model_name=self.embedding_details["model_name"],
                api_key=api_key,
                base_url=base_url,
            )

        if self.embedding_model in {"qwen", "embeddinggemma"}:
            return HuggingFaceEmbeddingFunction(
                model_name=self.embedding_details["model_name"]
            )

        if self.embedding_model not in ["default", "openai", "gemini"]:
            # Treat any other value as a HuggingFace model name
            return HuggingFaceEmbeddingFunction(
                model_name=self.embedding_details["model_name"]
            )

        # Use ChromaDB's default embedding function (all-MiniLM-L6-v2)
        return chromadb.utils.embedding_functions.DefaultEmbeddingFunction()

    def add_documents(
        self, documents: list[str], metadatas: list[dict[str, Any]], ids: list[str]
    ) -> None:
        """
        Add documents to the collection.

        Args:
            documents: List of document texts to embed
            metadatas: List of metadata dictionaries for each document
            ids: List of unique IDs for each document
        """
        try:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"Added {len(documents)} documents to ChromaDB collection")
        except Exception as e:
            if self._handle_runtime_embedding_mismatch(e, "add"):
                self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
                logger.info(
                    "Added %s documents to ChromaDB collection after reset",
                    len(documents),
                )
                return

            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise

    def upsert_documents(
        self, documents: list[str], metadatas: list[dict[str, Any]], ids: list[str]
    ) -> None:
        """
        Upsert (update or insert) documents to the collection.

        Args:
            documents: List of document texts to embed
            metadatas: List of metadata dictionaries for each document
            ids: List of unique IDs for each document
        """
        try:
            self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"Upserted {len(documents)} documents to ChromaDB collection")
        except Exception as e:
            if self._handle_runtime_embedding_mismatch(e, "upsert"):
                self.collection.upsert(
                    documents=documents, metadatas=metadatas, ids=ids
                )
                logger.info(
                    "Upserted %s documents to ChromaDB collection after reset",
                    len(documents),
                )
                return

            logger.error(f"Error upserting documents to ChromaDB: {e}")
            raise

    def search(
        self,
        query_texts: list[str],
        n_results: int = 10,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Search for similar documents.

        Args:
            query_texts: List of query texts
            n_results: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter conditions

        Returns:
            Search results from ChromaDB
        """
        try:
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document,
            )
            logger.info(
                f"Semantic search returned {len(results.get('ids', [[]])[0])} results"
            )
            return results
        except Exception as e:
            if self._handle_runtime_embedding_mismatch(e, "query"):
                results = self.collection.query(
                    query_texts=query_texts,
                    n_results=n_results,
                    where=where,
                    where_document=where_document,
                )
                logger.info(
                    "Semantic search returned %s results after reset",
                    len(results.get("ids", [[]])[0]),
                )
                return results

            logger.error(f"Error performing semantic search: {e}")
            raise

    def delete_documents(self, ids: list[str]) -> None:
        """
        Delete documents from the collection.

        Args:
            ids: List of document IDs to delete
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from ChromaDB collection")
        except Exception as e:
            logger.error(f"Error deleting documents from ChromaDB: {e}")
            raise

    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the collection."""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "embedding_model": self.embedding_model,
                "embedding_provider": self.embedding_details["provider"],
                "embedding_model_name": self.embedding_details["model_name"],
                "embedding_mismatch_behavior": self.embedding_mismatch_behavior,
                "persist_directory": self.persist_directory,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                "name": self.collection_name,
                "count": 0,
                "embedding_model": self.embedding_model,
                "embedding_provider": self.embedding_details["provider"],
                "embedding_model_name": self.embedding_details["model_name"],
                "embedding_mismatch_behavior": self.embedding_mismatch_behavior,
                "persist_directory": self.persist_directory,
                "error": str(e),
            }

    def reset_collection(self) -> None:
        """Reset (clear) the collection."""
        try:
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                # Collection may not exist yet.
                pass

            self.collection = self._create_collection()
            logger.info(
                "Reset ChromaDB collection '%s' with embedding %s:%s",
                self.collection_name,
                self.embedding_details["provider"],
                self.embedding_details["model_name"],
            )
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            raise

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists in the collection."""
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result["ids"]) > 0
        except Exception:
            return False

    def get_document_metadata(self, doc_id: str) -> dict[str, Any] | None:
        """
        Get metadata for a document if it exists.

        Args:
            doc_id: Document ID to look up

        Returns:
            Metadata dictionary if document exists, None otherwise
        """
        try:
            result = self.collection.get(ids=[doc_id], include=["metadatas"])
            if result["ids"] and result["metadatas"]:
                return result["metadatas"][0]
            return None
        except Exception:
            return None


def load_chroma_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load ChromaDB configuration from file and environment variables.

    Args:
        config_path: Path to configuration file

    Returns:
        Resolved ChromaDB configuration dictionary
    """
    # Default configuration
    config: dict[str, Any] = {
        "collection_name": "zotero_library",
        "embedding_model": "default",
        "embedding_config": {},
        "embedding_mismatch_behavior": "reset",
    }

    # Load configuration from file if it exists
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path) as f:
                file_config = json.load(f)
                config.update(file_config.get("semantic_search", {}))
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {e}")

    # Load configuration from environment variables
    env_embedding_model = os.getenv("ZOTERO_EMBEDDING_MODEL")
    if env_embedding_model:
        config["embedding_model"] = env_embedding_model

    env_mismatch_behavior = os.getenv("ZOTERO_EMBEDDING_MISMATCH_BEHAVIOR")
    if env_mismatch_behavior:
        config["embedding_mismatch_behavior"] = env_mismatch_behavior

    # Set up embedding config from environment
    if config["embedding_model"] == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        openai_base_url = os.getenv("OPENAI_BASE_URL")
        if openai_api_key:
            config["embedding_config"] = {
                "api_key": openai_api_key,
                "model_name": openai_model,
            }
            if openai_base_url:
                config["embedding_config"]["base_url"] = openai_base_url

    elif config["embedding_model"] == "gemini":
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        gemini_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
        gemini_base_url = os.getenv("GEMINI_BASE_URL")
        if gemini_api_key:
            config["embedding_config"] = {
                "api_key": gemini_api_key,
                "model_name": gemini_model,
            }
            if gemini_base_url:
                config["embedding_config"]["base_url"] = gemini_base_url

    return config


def create_chroma_client(config_path: str | None = None) -> ChromaClient:
    """
    Create a ChromaClient instance from configuration.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured ChromaClient instance
    """
    config = load_chroma_config(config_path)

    return ChromaClient(
        collection_name=config["collection_name"],
        embedding_model=config["embedding_model"],
        embedding_config=config["embedding_config"],
        embedding_mismatch_behavior=config["embedding_mismatch_behavior"],
    )
