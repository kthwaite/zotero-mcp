"""
Command-line interface for Zotero MCP server.
"""

import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

import typer

from zotero_mcp.server import mcp

app = typer.Typer(
    help="Zotero Model Context Protocol server",
    no_args_is_help=False,
    add_completion=False,
)

server_app = typer.Typer(help="Run the MCP server.")
config_app = typer.Typer(help="Initialize and inspect client configuration.")
index_app = typer.Typer(help="Manage semantic search index state.")
embeddings_app = typer.Typer(help="Warm embedding model caches.")
self_app = typer.Typer(help="Manage zotero-mcp installation metadata.")

app.add_typer(server_app, name="server")
app.add_typer(config_app, name="config")
app.add_typer(index_app, name="index")
app.add_typer(embeddings_app, name="embeddings")
app.add_typer(self_app, name="self")


def _warn_deprecated_command(old_command: str, new_command: str) -> None:
    """Emit a warning for a deprecated command alias."""
    typer.secho(
        f"Warning: '{old_command}' is deprecated; use '{new_command}' instead.",
        fg=typer.colors.YELLOW,
        err=True,
    )


def obfuscate_sensitive_value(value, keep_chars=4):
    """Obfuscate sensitive values by showing only the first few characters."""
    if not value or not isinstance(value, str):
        return value
    if len(value) <= keep_chars:
        return "*" * len(value)
    return value[:keep_chars] + "*" * (len(value) - keep_chars)


def obfuscate_config_for_display(config):
    """Create a copy of config with sensitive values obfuscated."""
    if not isinstance(config, dict):
        return config

    obfuscated = config.copy()
    sensitive_keys = ["ZOTERO_API_KEY", "ZOTERO_LIBRARY_ID", "API_KEY", "LIBRARY_ID"]

    for key in sensitive_keys:
        if key in obfuscated:
            obfuscated[key] = obfuscate_sensitive_value(obfuscated[key])

    return obfuscated


def load_claude_desktop_env_vars():
    """Load Zotero environment variables from Claude Desktop config unless globally disabled."""
    # Global guard to skip Claude detection entirely
    if str(os.environ.get("ZOTERO_NO_CLAUDE", "")).lower() in ("1", "true", "yes"):
        return {}
    from zotero_mcp.setup_helper import find_claude_config

    try:
        config_path = find_claude_config()
        if not config_path or not config_path.exists():
            return {}

        with open(config_path) as f:
            config = json.load(f)

        # Extract Zotero MCP server environment variables
        mcp_servers = config.get("mcpServers", {})
        zotero_config = mcp_servers.get("zotero", {})
        env_vars = zotero_config.get("env", {})

        return env_vars

    except Exception:
        return {}


def load_standalone_env_vars():
    """Load environment variables from standalone config (~/.config/zotero-mcp/config.json)."""
    try:
        cfg_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
        if not cfg_path.exists():
            return {}
        with open(cfg_path) as f:
            cfg = json.load(f)
        return cfg.get("client_env", {}) or {}
    except Exception:
        return {}


def apply_environment_variables(env_vars):
    """Apply environment variables to current process."""
    for key, value in env_vars.items():
        if key not in os.environ:  # Don't override existing env vars
            os.environ[key] = str(value)


def _save_zotero_db_path_to_config(config_path: Path, db_path: str) -> None:
    """
    Save the Zotero database path to the configuration file.

    This allows users to specify --db-path once and have it remembered
    for subsequent runs without needing to specify it again.

    Args:
        config_path: Path to the configuration file
        db_path: Path to the Zotero database file
    """
    try:
        # Ensure config directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing config or create new one
        full_config = {}
        if config_path.exists():
            try:
                with open(config_path) as f:
                    full_config = json.load(f)
            except Exception:
                pass

        # Ensure semantic_search section exists
        if "semantic_search" not in full_config:
            full_config["semantic_search"] = {}

        # Save the db_path
        full_config["semantic_search"]["zotero_db_path"] = db_path

        # Write back to file
        with open(config_path, "w") as f:
            json.dump(full_config, f, indent=2)

        print(f"Saved Zotero database path to config: {config_path}")

    except Exception as e:
        print(f"Warning: Could not save db_path to config: {e}")


def setup_zotero_environment():
    """Setup Zotero environment for CLI commands."""
    # Load standalone env first so global flags (e.g., ZOTERO_NO_CLAUDE) take effect
    standalone_env_vars = load_standalone_env_vars()
    apply_environment_variables(standalone_env_vars)

    # Respect global switch to disable Claude detection
    no_claude = str(os.environ.get("ZOTERO_NO_CLAUDE", "")).lower() in (
        "1",
        "true",
        "yes",
    )

    # Load and apply Claude Desktop env unless disabled
    if not no_claude:
        claude_env_vars = load_claude_desktop_env_vars()
        apply_environment_variables(claude_env_vars)

    # Apply fallback defaults for local Zotero if no config found
    fallback_env_vars = {
        "ZOTERO_LOCAL": "true",
        "ZOTERO_LIBRARY_ID": "0",
    }
    # Apply fallbacks only if not already set
    apply_environment_variables(fallback_env_vars)


def _run_serve(
    transport: Literal["stdio", "streamable-http", "sse"], host: str, port: int
) -> None:
    """Run the MCP server with the selected transport."""
    # Ensure environment is initialized (Claude config or standalone config)
    setup_zotero_environment()

    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "streamable-http":
        mcp.run(transport="streamable-http", host=host, port=port)
    elif transport == "sse":
        import warnings

        warnings.warn(
            "The SSE transport is deprecated and may be removed in a future version. "
            "New applications should use Streamable HTTP transport instead.",
            UserWarning,
        )
        mcp.run(transport="sse", host=host, port=port)


@app.callback(invoke_without_command=True)
def _default_command(ctx: typer.Context) -> None:
    """Run zotero-mcp commands."""
    if ctx.invoked_subcommand is None:
        _run_serve(transport="stdio", host="localhost", port=8000)


def _run_config_init(
    *,
    mode: Literal["local", "web"],
    client_target: Literal["auto", "claude", "standalone"],
    api_key: str | None,
    library_id: str | None,
    library_type: Literal["user", "group"],
    config: str | None,
    skip_semantic_search: bool,
    semantic_config_only: bool,
    embedding_backend: Literal[
        "minilm", "qwen", "embeddinggemma", "custom-hf", "openai", "gemini"
    ]
    | None,
    embedding_model: str | None,
    tui: bool,
) -> None:
    """Run setup workflow with normalized option names."""
    from zotero_mcp.setup_helper import run_setup

    no_local = mode == "web"
    no_claude = client_target == "standalone"

    if no_claude and config:
        print("Note: --config is ignored when --client-target=standalone.")

    if embedding_backend == "custom-hf" and not embedding_model:
        print("Error: --embedding-model is required when --embedding-backend=custom-hf")
        raise typer.Exit(code=1)

    exit_code = run_setup(
        no_local=no_local,
        no_claude=no_claude,
        api_key=api_key,
        library_id=library_id,
        library_type=library_type,
        config_path=None if no_claude else config,
        skip_semantic_search=skip_semantic_search,
        semantic_config_only=semantic_config_only,
        embedding_model=embedding_backend,
        embedding_model_name=embedding_model,
        tui=tui,
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@server_app.command("start")
def server_start(
    transport: Annotated[
        Literal["stdio", "streamable-http", "sse"],
        typer.Option(help="Transport to use (default: stdio)"),
    ] = "stdio",
    host: Annotated[
        str,
        typer.Option(
            help="Host to bind to for SSE/streamable-http transport (default: localhost)"
        ),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option(
            help="Port to bind to for SSE/streamable-http transport (default: 8000)"
        ),
    ] = 8000,
) -> None:
    """Run the MCP server."""
    _run_serve(transport=transport, host=host, port=port)


@app.command(hidden=True)
def serve(
    transport: Annotated[
        Literal["stdio", "streamable-http", "sse"],
        typer.Option(help="Transport to use (default: stdio)"),
    ] = "stdio",
    host: Annotated[
        str,
        typer.Option(
            help="Host to bind to for SSE/streamable-http transport (default: localhost)"
        ),
    ] = "localhost",
    port: Annotated[
        int,
        typer.Option(
            help="Port to bind to for SSE/streamable-http transport (default: 8000)"
        ),
    ] = 8000,
) -> None:
    """Deprecated alias for `server start`."""
    _warn_deprecated_command("zotero-mcp serve", "zotero-mcp server start")
    _run_serve(transport=transport, host=host, port=port)


@config_app.command("init")
def config_init(
    mode: Annotated[
        Literal["local", "web"],
        typer.Option(
            "--mode",
            help="Zotero connection mode (default: local)",
        ),
    ] = "local",
    client_target: Annotated[
        Literal["auto", "claude", "standalone"],
        typer.Option(
            "--client-target",
            help="Config destination (default: auto -> Claude Desktop config)",
        ),
    ] = "auto",
    api_key: Annotated[
        str | None,
        typer.Option(help="Zotero API key (needed for --mode web)"),
    ] = None,
    library_id: Annotated[
        str | None,
        typer.Option(help="Zotero library ID (needed for --mode web)"),
    ] = None,
    library_type: Annotated[
        Literal["user", "group"],
        typer.Option(help="Zotero library type (needed for --mode web)"),
    ] = "user",
    config: Annotated[
        str | None,
        typer.Option("--config", help="Path to Claude Desktop config file"),
    ] = None,
    skip_semantic_search: Annotated[
        bool,
        typer.Option(
            "--skip-semantic-search", help="Skip semantic search configuration"
        ),
    ] = False,
    semantic_config_only: Annotated[
        bool,
        typer.Option(
            "--semantic-config-only",
            help="Only configure semantic search, skip Zotero setup",
        ),
    ] = False,
    embedding_backend: Annotated[
        Literal["minilm", "qwen", "embeddinggemma", "custom-hf", "openai", "gemini"]
        | None,
        typer.Option(
            "--embedding-backend",
            help="Embedding backend for semantic search",
        ),
    ] = None,
    embedding_model: Annotated[
        str | None,
        typer.Option(
            "--embedding-model",
            help="Optional model override (or required HF model ID for custom-hf)",
        ),
    ] = None,
    tui: Annotated[
        bool,
        typer.Option("--tui", help="Launch full-screen Textual setup wizard"),
    ] = False,
) -> None:
    """Initialize zotero-mcp client and semantic-search configuration."""
    _run_config_init(
        mode=mode,
        client_target=client_target,
        api_key=api_key,
        library_id=library_id,
        library_type=library_type,
        config=config,
        skip_semantic_search=skip_semantic_search,
        semantic_config_only=semantic_config_only,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        tui=tui,
    )


@app.command(hidden=True)
def setup(
    no_local: Annotated[
        bool,
        typer.Option(
            "--no-local", help="Configure for Zotero Web API instead of local API"
        ),
    ] = False,
    api_key: Annotated[
        str | None,
        typer.Option(help="Zotero API key (only needed with --no-local)"),
    ] = None,
    library_id: Annotated[
        str | None,
        typer.Option(help="Zotero library ID (only needed with --no-local)"),
    ] = None,
    library_type: Annotated[
        Literal["user", "group"],
        typer.Option(help="Zotero library type (only needed with --no-local)"),
    ] = "user",
    no_claude: Annotated[
        bool,
        typer.Option(
            "--no-claude",
            help="Skip Claude Desktop config; write standalone config for web-based clients",
        ),
    ] = False,
    config_path: Annotated[
        str | None,
        typer.Option(help="Path to Claude Desktop config file"),
    ] = None,
    skip_semantic_search: Annotated[
        bool,
        typer.Option(
            "--skip-semantic-search", help="Skip semantic search configuration"
        ),
    ] = False,
    semantic_config_only: Annotated[
        bool,
        typer.Option(
            "--semantic-config-only",
            help="Only configure semantic search, skip Zotero setup",
        ),
    ] = False,
    embedding_model: Annotated[
        Literal["minilm", "qwen", "embeddinggemma", "custom-hf", "openai", "gemini"]
        | None,
        typer.Option(
            "--embedding-model",
            help="Embedding backend for semantic search (minilm, qwen, embeddinggemma, custom-hf, openai, gemini)",
        ),
    ] = None,
    embedding_model_name: Annotated[
        str | None,
        typer.Option(
            "--embedding-model-name",
            help="Optional model name override (for qwen/embeddinggemma/openai/gemini) or custom HF model ID",
        ),
    ] = None,
    tui: Annotated[
        bool,
        typer.Option("--tui", help="Launch full-screen Textual setup wizard"),
    ] = False,
) -> None:
    """Deprecated alias for `config init`."""
    _warn_deprecated_command("zotero-mcp setup", "zotero-mcp config init")
    _run_config_init(
        mode="web" if no_local else "local",
        client_target="standalone" if no_claude else "auto",
        api_key=api_key,
        library_id=library_id,
        library_type=library_type,
        config=config_path,
        skip_semantic_search=skip_semantic_search,
        semantic_config_only=semantic_config_only,
        embedding_backend=embedding_model,
        embedding_model=embedding_model_name,
        tui=tui,
    )


def _run_index_sync(
    force_rebuild: bool,
    limit: int | None,
    fulltext: bool,
    config: str | None,
    db_path: str | None,
) -> None:
    """Update semantic search database."""
    setup_zotero_environment()

    from zotero_mcp.semantic_search import create_semantic_search

    # Determine config path
    resolved_config_path = (
        Path(config)
        if config
        else Path.home() / ".config" / "zotero-mcp" / "config.json"
    )

    print(f"Using configuration: {resolved_config_path}")

    # Get optional db_path override from CLI
    if db_path:
        print(f"Using custom Zotero database: {db_path}")
        # Save the db_path to config file for future use
        _save_zotero_db_path_to_config(resolved_config_path, db_path)

    try:
        # Create semantic search instance with optional db_path override
        search = create_semantic_search(str(resolved_config_path), db_path=db_path)

        print("Starting database update...")
        if fulltext:
            print(
                "Note: --fulltext flag enabled. Will extract content from local database if available."
            )
        stats = search.update_database(
            force_full_rebuild=force_rebuild,
            limit=limit,
            extract_fulltext=fulltext,
        )

        print("\nDatabase update completed:")
        print(f"- Total items: {stats.get('total_items', 0)}")
        print(f"- Processed: {stats.get('processed_items', 0)}")
        print(f"- Added: {stats.get('added_items', 0)}")
        print(f"- Updated: {stats.get('updated_items', 0)}")
        print(f"- Skipped: {stats.get('skipped_items', 0)}")
        print(f"- Errors: {stats.get('errors', 0)}")
        print(f"- Duration: {stats.get('duration', 'Unknown')}")

        if stats.get("error"):
            print(f"Error: {stats['error']}")
            raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except Exception as e:
        print(f"Error updating database: {e}")
        raise typer.Exit(code=1)


@index_app.command("sync")
def index_sync(
    force_rebuild: Annotated[
        bool,
        typer.Option("--force-rebuild", help="Force complete rebuild of the database"),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option(help="Limit number of items to process (for testing)"),
    ] = None,
    fulltext: Annotated[
        bool,
        typer.Option(
            "--fulltext",
            help="Extract fulltext content from local Zotero database (slower but more comprehensive)",
        ),
    ] = False,
    config: Annotated[
        str | None,
        typer.Option("--config", help="Path to semantic search configuration file"),
    ] = None,
    db_path: Annotated[
        str | None,
        typer.Option(
            help="Path to Zotero database file (zotero.sqlite), overrides config"
        ),
    ] = None,
) -> None:
    """Update semantic search index."""
    _run_index_sync(
        force_rebuild=force_rebuild,
        limit=limit,
        fulltext=fulltext,
        config=config,
        db_path=db_path,
    )


@app.command("update-db", hidden=True)
def update_db(
    force_rebuild: Annotated[
        bool,
        typer.Option("--force-rebuild", help="Force complete rebuild of the database"),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option(help="Limit number of items to process (for testing)"),
    ] = None,
    fulltext: Annotated[
        bool,
        typer.Option(
            "--fulltext",
            help="Extract fulltext content from local Zotero database (slower but more comprehensive)",
        ),
    ] = False,
    config_path: Annotated[
        str | None,
        typer.Option(help="Path to semantic search configuration file"),
    ] = None,
    db_path: Annotated[
        str | None,
        typer.Option(
            help="Path to Zotero database file (zotero.sqlite), overrides config"
        ),
    ] = None,
) -> None:
    """Deprecated alias for `index sync`."""
    _warn_deprecated_command("zotero-mcp update-db", "zotero-mcp index sync")
    _run_index_sync(
        force_rebuild=force_rebuild,
        limit=limit,
        fulltext=fulltext,
        config=config_path,
        db_path=db_path,
    )


def _run_embeddings_warmup(
    embedding_backend: str | None,
    embedding_model: str | None,
    config: str | None,
) -> None:
    """Pre-download local embedding weights to avoid first-run startup lag."""
    setup_zotero_environment()

    import chromadb

    from zotero_mcp.chroma_client import (
        HuggingFaceEmbeddingFunction,
        load_chroma_config,
    )

    # Determine config path
    resolved_config_path = (
        Path(config)
        if config
        else Path.home() / ".config" / "zotero-mcp" / "config.json"
    )

    try:
        config_data = load_chroma_config(str(resolved_config_path))

        resolved_model = str(config_data.get("embedding_model", "default"))
        configured_model = resolved_model
        resolved_embedding_config = dict(config_data.get("embedding_config", {}) or {})

        cli_backend = embedding_backend.strip() if embedding_backend else None
        cli_model = embedding_model.strip() if embedding_model else None

        # Apply CLI overrides
        if cli_backend:
            if cli_backend == "minilm":
                resolved_model = "default"
                resolved_embedding_config = {}
            elif cli_backend == "custom-hf":
                if not cli_model:
                    print(
                        "Error: --embedding-model is required when --embedding-backend=custom-hf"
                    )
                    raise typer.Exit(code=1)
                resolved_model = cli_model
                resolved_embedding_config = {}
            else:
                resolved_model = cli_backend
                if resolved_model != configured_model:
                    resolved_embedding_config = {}

        # Optional model_name override for known model families
        if cli_model and cli_backend != "custom-hf":
            if resolved_model in {"qwen", "embeddinggemma", "openai", "gemini"}:
                resolved_embedding_config["model_name"] = cli_model
            elif not cli_backend:
                print(
                    "Note: --embedding-model was ignored because the configured "
                    f"embedding backend ('{resolved_model}') does not use model overrides."
                )
            else:
                print(
                    "Note: --embedding-model was ignored for "
                    f"embedding backend '{resolved_model}'."
                )

        if resolved_model == "openai":
            model_name = resolved_embedding_config.get(
                "model_name", "text-embedding-3-small"
            )
            print(
                f"Embedding backend is OpenAI ({model_name}); no local weights to download."
            )
            return

        if resolved_model == "gemini":
            model_name = resolved_embedding_config.get(
                "model_name", "models/text-embedding-004"
            )
            print(
                f"Embedding backend is Gemini ({model_name}); no local weights to download."
            )
            return

        if resolved_model == "default":
            print("Warming default MiniLM embedding model cache (all-MiniLM-L6-v2)...")
            embedding_function = (
                chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
            )
            embedding_function(["zotero-mcp embedding warmup"])
            print("✅ Default embedding model is ready.")
            return

        if resolved_model == "qwen":
            hf_model_name = resolved_embedding_config.get(
                "model_name", "Qwen/Qwen3-Embedding-0.6B"
            )
        elif resolved_model == "embeddinggemma":
            hf_model_name = resolved_embedding_config.get(
                "model_name", "google/embeddinggemma-300m"
            )
        else:
            hf_model_name = resolved_model

        print(f"Warming HuggingFace embedding model cache: {hf_model_name}")
        embedding_function = HuggingFaceEmbeddingFunction(model_name=hf_model_name)
        embedding_function(["zotero-mcp embedding warmup"])
        print(f"✅ Embedding model is ready: {hf_model_name}")

    except typer.Exit:
        raise
    except Exception as e:
        print(f"Error pre-downloading embedding model: {e}")
        raise typer.Exit(code=1)


@embeddings_app.command("warmup")
def embeddings_warmup(
    embedding_backend: Annotated[
        str | None,
        typer.Option(
            "--embedding-backend",
            help="Embedding backend to warm up (minilm, qwen, embeddinggemma, custom-hf, openai, gemini)",
        ),
    ] = None,
    embedding_model: Annotated[
        str | None,
        typer.Option(
            "--embedding-model",
            help="Optional model override (or required HF model ID for custom-hf)",
        ),
    ] = None,
    config: Annotated[
        str | None,
        typer.Option("--config", help="Path to semantic search configuration file"),
    ] = None,
) -> None:
    """Pre-download local embedding model weights."""
    _run_embeddings_warmup(
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        config=config,
    )


@app.command("download-embeddings", hidden=True)
def download_embeddings(
    embedding_model: Annotated[
        str | None,
        typer.Option(
            "--embedding-model",
            help="Embedding model/backend to warm up (minilm, qwen, embeddinggemma, custom-hf, openai, gemini, or a HuggingFace model ID)",
        ),
    ] = None,
    embedding_model_name: Annotated[
        str | None,
        typer.Option(
            "--embedding-model-name",
            help="Optional model name override (for qwen/embeddinggemma/openai/gemini) or custom HF model ID when --embedding-model=custom-hf",
        ),
    ] = None,
    config_path: Annotated[
        str | None,
        typer.Option(help="Path to semantic search configuration file"),
    ] = None,
) -> None:
    """Deprecated alias for `embeddings warmup`."""
    _warn_deprecated_command(
        "zotero-mcp download-embeddings", "zotero-mcp embeddings warmup"
    )
    _run_embeddings_warmup(
        embedding_backend=embedding_model,
        embedding_model=embedding_model_name,
        config=config_path,
    )


def _run_index_status(config: str | None) -> None:
    """Show semantic search database status."""
    setup_zotero_environment()

    from zotero_mcp.semantic_search import create_semantic_search

    # Determine config path
    resolved_config_path = (
        Path(config)
        if config
        else Path.home() / ".config" / "zotero-mcp" / "config.json"
    )

    try:
        # Create semantic search instance
        search = create_semantic_search(str(resolved_config_path))

        # Get database status
        status = search.get_database_status()

        print("=== Semantic Search Database Status ===")

        collection_info = status.get("collection_info", {})
        print(f"Collection: {collection_info.get('name', 'Unknown')}")
        print(f"Document count: {collection_info.get('count', 0)}")
        print(f"Embedding model: {collection_info.get('embedding_model', 'Unknown')}")
        print(f"Database path: {collection_info.get('persist_directory', 'Unknown')}")

        update_config = status.get("update_config", {})
        print("\nUpdate configuration:")
        print(f"- Auto update: {update_config.get('auto_update', False)}")
        print(f"- Frequency: {update_config.get('update_frequency', 'manual')}")
        print(f"- Last update: {update_config.get('last_update', 'Never')}")
        print(f"- Should update: {status.get('should_update', False)}")

        if collection_info.get("error"):
            print(f"\nError: {collection_info['error']}")

    except Exception as e:
        print(f"Error getting database status: {e}")
        raise typer.Exit(code=1)


@index_app.command("status")
def index_status(
    config: Annotated[
        str | None,
        typer.Option("--config", help="Path to semantic search configuration file"),
    ] = None,
) -> None:
    """Show semantic search index status."""
    _run_index_status(config=config)


@app.command("db-status", hidden=True)
def db_status(
    config_path: Annotated[
        str | None,
        typer.Option(help="Path to semantic search configuration file"),
    ] = None,
) -> None:
    """Deprecated alias for `index status`."""
    _warn_deprecated_command("zotero-mcp db-status", "zotero-mcp index status")
    _run_index_status(config=config_path)


def _run_index_inspect(
    limit: int,
    filter_text: str | None,
    show_documents: bool,
    stats: bool,
    config: str | None,
) -> None:
    """Inspect indexed documents or show aggregate stats for the semantic DB."""
    setup_zotero_environment()

    from zotero_mcp.semantic_search import create_semantic_search

    # Determine config path
    resolved_config_path = (
        Path(config)
        if config
        else Path.home() / ".config" / "zotero-mcp" / "config.json"
    )

    try:
        search = create_semantic_search(str(resolved_config_path))
        client = search.chroma_client
        col = client.collection

        if stats:
            # Show aggregate stats (merged from former db-stats)
            meta = col.get(include=["metadatas"])
            metas = meta.get("metadatas", [])
            print("=== Semantic DB Inspection (Stats) ===")
            info = client.get_collection_info()
            print(f"Collection: {info.get('name')} @ {info.get('persist_directory')}")
            print(f"Count: {info.get('count')}")

            # Item type distribution
            item_types = [(m or {}).get("item_type", "") for m in metas]
            ct_types = Counter(item_types)
            print("Item types:")
            for t, c in ct_types.most_common(20):
                print(f"  {t or '(missing)'}: {c}")

            # Fulltext coverage by type (pdf/html)
            coverage = {}
            for m in metas:
                m = m or {}
                t = m.get("item_type", "") or "(missing)"
                cov = coverage.setdefault(
                    t, {"total": 0, "with_fulltext": 0, "pdf": 0, "html": 0}
                )
                cov["total"] += 1
                if m.get("has_fulltext"):
                    cov["with_fulltext"] += 1
                    src = (m.get("fulltext_source") or "").lower()
                    if src == "pdf":
                        cov["pdf"] += 1
                    elif src == "html":
                        cov["html"] += 1
            print("Fulltext coverage (by type):")
            for t, cov in coverage.items():
                print(
                    f"  {t}: {cov['with_fulltext']}/{cov['total']} (pdf:{cov['pdf']}, html:{cov['html']})"
                )

            # Common titles (may indicate duplicates)
            titles = [(m or {}).get("title", "") for m in metas]
            ct_titles = Counter([t for t in titles if t])
            common = [(t, c) for t, c in ct_titles.most_common(10)]
            if common:
                print("Common titles:")
                for t, c in common:
                    print(f"  {t[:80]}{'...' if len(t) > 80 else ''}: {c}")
            return

        include = ["metadatas"]
        if show_documents:
            include.append("documents")

        # Fetch up to limit; filter client-side if requested
        data = col.get(limit=limit, include=include)

        print("=== Semantic DB Inspection ===")
        total = client.get_collection_info().get("count", 0)
        print(f"Total documents: {total}")
        print(f"Showing up to: {limit}")

        shown = 0
        for i, meta in enumerate(data.get("metadatas", [])):
            meta = meta or {}
            title = meta.get("title", "")
            creators = meta.get("creators", "")
            if filter_text:
                needle = filter_text.lower()
                if (
                    needle not in (title or "").lower()
                    and needle not in (creators or "").lower()
                ):
                    continue
            print(f"- {title} | {creators}")
            if show_documents:
                doc = (data.get("documents", [""])[i] or "").strip()
                snippet = doc[:200].replace("\n", " ") + (
                    "..." if len(doc) > 200 else ""
                )
                if snippet:
                    print(f"  doc: {snippet}")
            shown += 1
            if shown >= limit:
                break

        if shown == 0:
            print("No records matched your filter.")

    except Exception as e:
        print(f"Error inspecting database: {e}")
        raise typer.Exit(code=1)


@index_app.command("inspect")
def index_inspect(
    limit: Annotated[
        int,
        typer.Option(help="How many records to show (default: 20)"),
    ] = 20,
    filter_text: Annotated[
        str | None,
        typer.Option("--filter", help="Substring to match in title or creators"),
    ] = None,
    show_documents: Annotated[
        bool,
        typer.Option("--show-documents", help="Show beginning of stored document text"),
    ] = False,
    stats: Annotated[
        bool,
        typer.Option("--stats", help="Show aggregate stats (formerly db-stats)"),
    ] = False,
    config: Annotated[
        str | None,
        typer.Option("--config", help="Path to semantic search configuration file"),
    ] = None,
) -> None:
    """Inspect semantic search index contents."""
    _run_index_inspect(
        limit=limit,
        filter_text=filter_text,
        show_documents=show_documents,
        stats=stats,
        config=config,
    )


@app.command("db-inspect", hidden=True)
def db_inspect(
    limit: Annotated[
        int,
        typer.Option(help="How many records to show (default: 20)"),
    ] = 20,
    filter_text: Annotated[
        str | None,
        typer.Option("--filter", help="Substring to match in title or creators"),
    ] = None,
    show_documents: Annotated[
        bool,
        typer.Option("--show-documents", help="Show beginning of stored document text"),
    ] = False,
    stats: Annotated[
        bool,
        typer.Option("--stats", help="Show aggregate stats (formerly db-stats)"),
    ] = False,
    config_path: Annotated[
        str | None,
        typer.Option(help="Path to semantic search configuration file"),
    ] = None,
) -> None:
    """Deprecated alias for `index inspect`."""
    _warn_deprecated_command("zotero-mcp db-inspect", "zotero-mcp index inspect")
    _run_index_inspect(
        limit=limit,
        filter_text=filter_text,
        show_documents=show_documents,
        stats=stats,
        config=config_path,
    )


def _run_self_update(
    check_only: bool,
    force: bool,
    method: Literal["pip", "uv", "conda", "pipx"] | None,
) -> None:
    """Update zotero-mcp to the latest version."""
    from zotero_mcp.updater import update_zotero_mcp

    try:
        print("Checking for updates...")

        result = update_zotero_mcp(
            check_only=check_only,
            force=force,
            method=method,
        )

        print("\n" + "=" * 50)
        print("UPDATE RESULTS")
        print("=" * 50)

        if check_only:
            print(f"Current version: {result.get('current_version', 'Unknown')}")
            print(f"Latest version: {result.get('latest_version', 'Unknown')}")
            print(f"Update needed: {result.get('needs_update', False)}")
            print(f"Status: {result.get('message', 'Unknown')}")
        else:
            if result.get("success"):
                print("✅ Update completed successfully!")
                print(
                    f"Version: {result.get('current_version', 'Unknown')} → {result.get('latest_version', 'Unknown')}"
                )
                print(f"Method: {result.get('method', 'Unknown')}")
                print(f"Message: {result.get('message', '')}")

                print("\n📋 Next steps:")
                print("• All configurations have been preserved")
                print("• Restart Claude Desktop if it's running")
                print("• Your semantic search database is intact")
                print("• Run 'zotero-mcp self version' to verify the update")
            else:
                print("❌ Update failed!")
                print(f"Error: {result.get('message', 'Unknown error')}")

                if backup_dir := result.get("backup_dir"):
                    print(f"\n🔄 Backup created at: {backup_dir}")
                    print("You can manually restore configurations if needed")

                raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except Exception as e:
        print(f"❌ Update error: {e}")
        raise typer.Exit(code=1)


@self_app.command("update")
def self_update(
    check_only: Annotated[
        bool,
        typer.Option("--check-only", help="Only check for updates without installing"),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Force update even if already up to date"),
    ] = False,
    method: Annotated[
        Literal["pip", "uv", "conda", "pipx"] | None,
        typer.Option(help="Override auto-detected installation method"),
    ] = None,
) -> None:
    """Update zotero-mcp to the latest version."""
    _run_self_update(check_only=check_only, force=force, method=method)


@app.command(hidden=True)
def update(
    check_only: Annotated[
        bool,
        typer.Option("--check-only", help="Only check for updates without installing"),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Force update even if already up to date"),
    ] = False,
    method: Annotated[
        Literal["pip", "uv", "conda", "pipx"] | None,
        typer.Option(help="Override auto-detected installation method"),
    ] = None,
) -> None:
    """Deprecated alias for `self update`."""
    _warn_deprecated_command("zotero-mcp update", "zotero-mcp self update")
    _run_self_update(check_only=check_only, force=force, method=method)


def _run_self_version() -> None:
    """Print version information."""
    from zotero_mcp._version import __version__

    print(f"Zotero MCP v{__version__}")


@self_app.command("version")
def self_version() -> None:
    """Print version information."""
    _run_self_version()


@app.command(hidden=True)
def version() -> None:
    """Top-level version alias."""
    _run_self_version()


def _run_config_show() -> None:
    """Show installation path and configuration info for MCP clients."""
    # Setup Zotero environment variables
    setup_zotero_environment()

    # Get the installation path
    executable_path = shutil.which("zotero-mcp")
    if not executable_path:
        executable_path = sys.executable + " -m zotero_mcp"

    # Determine whether Claude is disabled globally
    no_claude = str(os.environ.get("ZOTERO_NO_CLAUDE", "")).lower() in (
        "1",
        "true",
        "yes",
    )

    # Load current environment configurations
    standalone_env_vars = load_standalone_env_vars()
    claude_env_vars = {} if no_claude else load_claude_desktop_env_vars()

    # Choose which env to display: prefer standalone if present or if Claude disabled
    display_env = (
        standalone_env_vars
        if (no_claude or standalone_env_vars)
        else (claude_env_vars or {"ZOTERO_LOCAL": "true"})
    )

    print("=== Zotero MCP Setup Information ===")
    print()
    print("🔧 Installation Details:")
    print(f"  Command path: {executable_path}")
    print(f"  Python path: {sys.executable}")

    # Detect installation method
    try:
        # Check if installed via uv
        result = subprocess.run(
            ["uv", "tool", "list"], capture_output=True, text=True, timeout=5
        )
        if "zotero-mcp" in result.stdout:
            print("  Installation method: uv tool")
        else:
            # Check pip
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", "zotero-mcp"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                print("  Installation method: pip")
            else:
                print("  Installation method: unknown")
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        print("  Installation method: unknown")

    print()
    print("⚙️  MCP Client Configuration:")
    print(f"  Command: {executable_path}")
    print("  Arguments: [] (empty)")

    # Show environment variables with obfuscated sensitive values
    obfuscated_env_vars = obfuscate_config_for_display(display_env)
    print(
        f"  Environment (single-line): {json.dumps(obfuscated_env_vars, separators=(',', ':'))}"
    )
    print(
        "  💡 Note: This shows client config. Shell variables may override for CLI use."
    )
    print(f"  Claude integration: {'disabled' if no_claude else 'enabled'}")

    # Only show Claude Desktop config if not globally disabled
    if not no_claude:
        print()
        print("For Claude Desktop (claude_desktop_config.json):")
        config_snippet = {
            "mcpServers": {
                "zotero": {
                    "command": executable_path,
                    "env": obfuscated_env_vars,
                }
            }
        }
        print(json.dumps(config_snippet, indent=2))

    # Show semantic search database info with detailed statistics
    print()
    print("🧠 Semantic Search Database:")

    # Check for semantic search config
    config_path = Path.home() / ".config" / "zotero-mcp" / "config.json"
    if config_path.exists():
        try:
            from zotero_mcp.semantic_search import create_semantic_search

            # Get database status (similar to index status command)
            search = create_semantic_search(str(config_path))
            status = search.get_database_status()

            collection_info = status.get("collection_info", {})

            print("  Status: ✅ Configuration file found")
            print(f"  Config path: {config_path}")
            print(f"  Collection: {collection_info.get('name', 'Unknown')}")
            print(f"  Document count: {collection_info.get('count', 0)}")
            print(
                f"  Embedding model: {collection_info.get('embedding_model', 'Unknown')}"
            )
            print(
                f"  Database path: {collection_info.get('persist_directory', 'Unknown')}"
            )

            update_config = status.get("update_config", {})
            print(f"  Auto update: {update_config.get('auto_update', False)}")
            print(
                f"  Update frequency: {update_config.get('update_frequency', 'manual')}"
            )
            print(f"  Last update: {update_config.get('last_update', 'Never')}")
            print(f"  Should update: {status.get('should_update', False)}")

            if collection_info.get("error"):
                print(f"  Error: {collection_info['error']}")

        except Exception as e:
            print("  Status: ⚠️ Configuration found but database error")
            print(f"  Error: {e}")
    else:
        print("  Status: ⚠️ Not configured")
        print("  💡 Run 'zotero-mcp config init' to configure semantic search")


@config_app.command("show")
def config_show() -> None:
    """Show installation and configuration info for MCP clients."""
    _run_config_show()


@app.command("setup-info", hidden=True)
def setup_info() -> None:
    """Deprecated alias for `config show`."""
    _warn_deprecated_command("zotero-mcp setup-info", "zotero-mcp config show")
    _run_config_show()


def main() -> None:
    """Main entry point for console scripts."""
    app()


if __name__ == "__main__":
    main()
