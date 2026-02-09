#!/usr/bin/env python

"""
Setup helper for zotero-mcp.

This script provides utilities to automatically configure zotero-mcp
by finding the installed executable and updating Claude Desktop's config.
"""

import getpass
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Annotated, Literal

import typer

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, IntPrompt, Prompt
    from rich.table import Table

    _RICH_AVAILABLE = True
except Exception:  # pragma: no cover - fallback only used when rich is unavailable
    _RICH_AVAILABLE = False


EmbeddingModelOption = Literal[
    "minilm", "qwen", "embeddinggemma", "custom-hf", "openai", "gemini"
]

_KNOWN_EMBEDDING_MODELS = {
    "default",
    "qwen",
    "embeddinggemma",
    "openai",
    "gemini",
}

_console = Console() if _RICH_AVAILABLE else None


def _ui_print(message: str) -> None:
    if _console:
        _console.print(message)
    else:
        print(message)


def _ui_section(title: str) -> None:
    if _console:
        _console.print(
            Panel.fit(f"[bold cyan]{title}[/bold cyan]", border_style="cyan")
        )
    else:
        print(f"\n=== {title} ===")


def _ui_confirm(prompt: str, default: bool = False) -> bool:
    if _console:
        return Confirm.ask(prompt, default=default)

    suffix = "[Y/n]" if default else "[y/N]"
    value = input(f"{prompt} {suffix}: ").strip().lower()
    if not value:
        return default
    return value in {"y", "yes"}


def _ui_prompt(
    prompt: str,
    default: str | None = None,
    show_default: bool = True,
) -> str:
    if _console:
        return Prompt.ask(prompt, default=default, show_default=show_default)

    suffix = f" [{default}]" if show_default and default not in (None, "") else ""
    value = input(f"{prompt}{suffix}: ").strip()
    if not value and default is not None:
        return default
    return value


def _ui_int_prompt(prompt: str, default: int | None = None, minimum: int = 1) -> int:
    if _console:
        while True:
            value = IntPrompt.ask(prompt, default=default)
            if value >= minimum:
                return value
            _ui_print(f"Please enter a number >= {minimum}")

    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw == "" and default is not None:
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                _ui_print("Please enter a valid number")
                continue
        if value >= minimum:
            return value
        _ui_print(f"Please enter a number >= {minimum}")


def find_executable():
    """Find the full path to the zotero-mcp executable."""
    # Try to find the executable in the PATH
    exe_name = "zotero-mcp"
    if sys.platform == "win32":
        exe_name += ".exe"

    exe_path = shutil.which(exe_name)
    if exe_path:
        print(f"Found zotero-mcp in PATH at: {exe_path}")
        return exe_path

    # If not found in PATH, try to find it in common installation directories
    potential_paths = []

    # User site-packages
    import site

    for site_path in site.getsitepackages():
        potential_paths.append(Path(site_path) / "bin" / exe_name)

    # User's home directory
    potential_paths.append(Path.home() / ".local" / "bin" / exe_name)

    # Virtual environment
    if "VIRTUAL_ENV" in os.environ:
        potential_paths.append(Path(os.environ["VIRTUAL_ENV"]) / "bin" / exe_name)

    # Additional common locations
    if sys.platform == "darwin":  # macOS
        potential_paths.append(Path("/usr/local/bin") / exe_name)
        potential_paths.append(Path("/opt/homebrew/bin") / exe_name)

    for path in potential_paths:
        if path.exists() and os.access(path, os.X_OK):
            print(f"Found zotero-mcp at: {path}")
            return str(path)

    # If still not found, search in common directories
    print("Searching for zotero-mcp in common locations...")
    try:
        # On Unix-like systems, try using the 'find' command
        if sys.platform != "win32":
            import subprocess

            result = subprocess.run(
                [
                    "find",
                    os.path.expanduser("~"),
                    "-name",
                    "zotero-mcp",
                    "-type",
                    "f",
                    "-executable",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            paths = result.stdout.strip().split("\n")
            if paths and paths[0]:
                print(f"Found zotero-mcp at {paths[0]}")
                return paths[0]
    except Exception as e:
        print(f"Error searching for zotero-mcp: {e}")

    print("Warning: Could not find zotero-mcp executable.")
    print("Make sure zotero-mcp is installed and in your PATH.")
    return None


def find_claude_config():
    """Find Claude Desktop config file path."""
    config_paths = []

    # macOS
    if sys.platform == "darwin":
        # Try both old and new paths
        config_paths.append(
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
        config_paths.append(
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude Desktop"
            / "claude_desktop_config.json"
        )

    # Windows
    elif sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            config_paths.append(Path(appdata) / "Claude" / "claude_desktop_config.json")
            config_paths.append(
                Path(appdata) / "Claude Desktop" / "claude_desktop_config.json"
            )

    # Linux
    else:
        config_home = os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
        config_paths.append(Path(config_home) / "Claude" / "claude_desktop_config.json")
        config_paths.append(
            Path(config_home) / "Claude Desktop" / "claude_desktop_config.json"
        )

    # Check all possible locations
    for path in config_paths:
        if path.exists():
            print(f"Found Claude Desktop config at: {path}")
            return path

    # Return the default path for the platform if not found
    # We'll use the newer "Claude Desktop" path as default
    if sys.platform == "darwin":  # macOS
        default_path = (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude Desktop"
            / "claude_desktop_config.json"
        )
    elif sys.platform == "win32":  # Windows
        appdata = os.environ.get("APPDATA", "")
        default_path = Path(appdata) / "Claude Desktop" / "claude_desktop_config.json"
    else:  # Linux and others
        config_home = os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
        default_path = (
            Path(config_home) / "Claude Desktop" / "claude_desktop_config.json"
        )

    print(f"Claude Desktop config not found. Using default path: {default_path}")
    return default_path


def setup_semantic_search(
    existing_semantic_config: dict | None = None,
    embedding_model: EmbeddingModelOption | None = None,
    embedding_model_name: str | None = None,
) -> dict:
    """Interactive setup for semantic search configuration."""
    _ui_section("Semantic Search Configuration")

    existing_semantic_config = existing_semantic_config or {}

    if existing_semantic_config:
        # Display config without sensitive info
        model = existing_semantic_config.get("embedding_model", "unknown")
        model_name = existing_semantic_config.get("embedding_config", {}).get(
            "model_name"
        )
        if not model_name and model not in _KNOWN_EMBEDDING_MODELS:
            model_name = model
        update_freq = existing_semantic_config.get("update_config", {}).get(
            "update_frequency", "unknown"
        )
        db_path = existing_semantic_config.get("zotero_db_path", "auto-detect")

        if _console:
            table = Table(show_header=False, box=None)
            table.add_row("Current embedding model", str(model))
            table.add_row("Current model name", str(model_name or "(default)"))
            table.add_row("Current update frequency", str(update_freq))
            table.add_row("Current Zotero database path", str(db_path))
            _console.print(table)
        else:
            _ui_print("Found existing semantic search configuration:")
            _ui_print(f"  - Embedding model: {model}")
            _ui_print(f"  - Embedding model name: {model_name or '(default)'}")
            _ui_print(f"  - Update frequency: {update_freq}")
            _ui_print(f"  - Zotero database path: {db_path}")

        if embedding_model is None:
            _ui_print(
                "You can keep this config or change it. If you change model settings, a DB rebuild is advised."
            )
            if _ui_confirm(
                "Keep existing semantic search configuration?", default=True
            ):
                return existing_semantic_config
        else:
            _ui_print("Embedding model override provided via CLI; reconfiguring.")

    _ui_print(
        "Configure embedding models for semantic search over your Zotero library."
    )

    selection_map = {
        "1": "minilm",
        "2": "qwen",
        "3": "embeddinggemma",
        "4": "custom-hf",
        "5": "openai",
        "6": "gemini",
    }

    selected_model = embedding_model
    if selected_model is None:
        if _console:
            table = Table(title="Available embedding models")
            table.add_column("#", style="cyan", no_wrap=True)
            table.add_column("Model", style="bold")
            table.add_column("Type")
            table.add_column("Notes")
            table.add_row("1", "MiniLM (default)", "Local", "all-MiniLM-L6-v2")
            table.add_row("2", "Qwen", "Local", "Qwen/Qwen3-Embedding-0.6B")
            table.add_row("3", "EmbeddingGemma", "Local", "google/embeddinggemma-300m")
            table.add_row("4", "Custom HuggingFace", "Local", "Any HF model ID")
            table.add_row("5", "OpenAI", "Remote", "Requires API key")
            table.add_row("6", "Gemini", "Remote", "Requires API key")
            _console.print(table)
        else:
            _ui_print("\nAvailable embedding models:")
            _ui_print("1. MiniLM (default) - local (all-MiniLM-L6-v2)")
            _ui_print("2. Qwen - local (Qwen/Qwen3-Embedding-0.6B)")
            _ui_print("3. EmbeddingGemma - local (google/embeddinggemma-300m)")
            _ui_print("4. Custom HuggingFace - local (any HF model ID)")
            _ui_print("5. OpenAI - remote (requires API key)")
            _ui_print("6. Gemini - remote (requires API key)")

        while True:
            choice = _ui_prompt("Choose embedding model (1-6)", default="1")
            if choice in selection_map:
                selected_model = selection_map[choice]
                break
            _ui_print("Please enter 1, 2, 3, 4, 5, or 6")

    if selected_model not in selection_map.values():
        raise ValueError(
            f"Unsupported embedding model: {selected_model}. "
            "Choose one of: minilm, qwen, embeddinggemma, custom-hf, openai, gemini"
        )

    config: dict[str, object] = {}

    if selected_model == "minilm":
        config["embedding_model"] = "default"
        if embedding_model_name:
            _ui_print("Note: --embedding-model-name is ignored for minilm.")
        _ui_print("Using local MiniLM embedding model (all-MiniLM-L6-v2)")

    elif selected_model == "qwen":
        config["embedding_model"] = "qwen"
        if embedding_model_name:
            config["embedding_config"] = {"model_name": embedding_model_name}
            _ui_print(f"Using local Qwen embedding model: {embedding_model_name}")
        else:
            _ui_print("Using local Qwen embedding model (Qwen/Qwen3-Embedding-0.6B)")

    elif selected_model == "embeddinggemma":
        config["embedding_model"] = "embeddinggemma"
        if embedding_model_name:
            config["embedding_config"] = {"model_name": embedding_model_name}
            _ui_print(
                f"Using local EmbeddingGemma embedding model: {embedding_model_name}"
            )
        else:
            _ui_print(
                "Using local EmbeddingGemma embedding model (google/embeddinggemma-300m)"
            )

    elif selected_model == "custom-hf":
        default_custom_hf = embedding_model_name
        if not default_custom_hf:
            existing_model = existing_semantic_config.get("embedding_model", "")
            if existing_model and existing_model not in _KNOWN_EMBEDDING_MODELS:
                default_custom_hf = existing_model

        custom_hf = embedding_model_name or _ui_prompt(
            "Enter HuggingFace model ID",
            default=default_custom_hf or None,
        )
        custom_hf = custom_hf.strip()
        if not custom_hf:
            raise ValueError(
                "Custom HuggingFace model ID is required when --embedding-model=custom-hf"
            )

        config["embedding_model"] = custom_hf
        _ui_print(f"Using custom local HuggingFace model: {custom_hf}")

    elif selected_model == "openai":
        config["embedding_model"] = "openai"

        if embedding_model_name:
            model_name = embedding_model_name
        else:
            _ui_print("\nOpenAI embedding models:")
            _ui_print("1. text-embedding-3-small (recommended, faster)")
            _ui_print("2. text-embedding-3-large (higher quality, slower)")
            while True:
                model_choice = _ui_prompt("Choose OpenAI model (1-2)", default="1")
                if model_choice in {"1", "2"}:
                    break
                _ui_print("Please enter 1 or 2")
            model_name = (
                "text-embedding-3-small"
                if model_choice == "1"
                else "text-embedding-3-large"
            )

        config["embedding_config"] = {"model_name": model_name}

        api_key = getpass.getpass("Enter your OpenAI API key (hidden): ").strip()
        if api_key:
            config["embedding_config"]["api_key"] = api_key
        else:
            _ui_print(
                "Warning: No API key provided. Set OPENAI_API_KEY environment variable."
            )

        base_url = _ui_prompt(
            "Enter custom OpenAI base URL (leave blank for default)",
            default="",
            show_default=False,
        ).strip()
        if base_url:
            config["embedding_config"]["base_url"] = base_url
            _ui_print(f"Using custom OpenAI base URL: {base_url}")
        else:
            _ui_print("Using default OpenAI base URL")

    elif selected_model == "gemini":
        config["embedding_model"] = "gemini"

        if embedding_model_name:
            model_name = embedding_model_name
        else:
            _ui_print("\nGemini embedding models:")
            _ui_print("1. models/text-embedding-004 (recommended)")
            _ui_print("2. models/gemini-embedding-exp-03-07 (experimental)")
            while True:
                model_choice = _ui_prompt("Choose Gemini model (1-2)", default="1")
                if model_choice in {"1", "2"}:
                    break
                _ui_print("Please enter 1 or 2")
            model_name = (
                "models/text-embedding-004"
                if model_choice == "1"
                else "models/gemini-embedding-exp-03-07"
            )

        config["embedding_config"] = {"model_name": model_name}

        api_key = getpass.getpass("Enter your Gemini API key (hidden): ").strip()
        if api_key:
            config["embedding_config"]["api_key"] = api_key
        else:
            _ui_print(
                "Warning: No API key provided. Set GEMINI_API_KEY environment variable."
            )

        base_url = _ui_prompt(
            "Enter custom Gemini base URL (leave blank for default)",
            default="",
            show_default=False,
        ).strip()
        if base_url:
            config["embedding_config"]["base_url"] = base_url
            _ui_print(f"Using custom Gemini base URL: {base_url}")
        else:
            _ui_print("Using default Gemini base URL")

    # Configure update frequency
    _ui_section("Database Update Configuration")
    _ui_print("Configure how often the semantic search database is updated:")
    _ui_print("1. Manual - Update only when you run 'zotero-mcp update-db'")
    _ui_print("2. Auto - Automatically update on server startup")
    _ui_print("3. Daily - Automatically update once per day")
    _ui_print("4. Every N days - Automatically update every N days")

    existing_update = existing_semantic_config.get("update_config", {})
    existing_frequency = existing_update.get("update_frequency", "manual")
    default_update_choice = "1"
    if existing_frequency == "startup":
        default_update_choice = "2"
    elif existing_frequency == "daily":
        default_update_choice = "3"
    elif str(existing_frequency).startswith("every_"):
        default_update_choice = "4"

    while True:
        update_choice = _ui_prompt(
            "Choose update frequency (1-4)", default=default_update_choice
        )
        if update_choice in {"1", "2", "3", "4"}:
            break
        _ui_print("Please enter 1, 2, 3, or 4")

    if update_choice == "1":
        update_config = {
            "auto_update": False,
            "update_frequency": "manual",
        }
        _ui_print("Database will only be updated manually.")
    elif update_choice == "2":
        update_config = {
            "auto_update": True,
            "update_frequency": "startup",
        }
        _ui_print("Database will be updated every time the server starts.")
    elif update_choice == "3":
        update_config = {
            "auto_update": True,
            "update_frequency": "daily",
        }
        _ui_print("Database will be updated once per day.")
    else:
        default_days = existing_update.get("update_days")
        if not isinstance(default_days, int) or default_days <= 0:
            if str(existing_frequency).startswith("every_"):
                try:
                    default_days = int(str(existing_frequency).split("_", 1)[1])
                except Exception:
                    default_days = 7
            else:
                default_days = 7

        days = _ui_int_prompt(
            "Enter number of days between updates", default=default_days
        )

        update_config = {
            "auto_update": True,
            "update_frequency": f"every_{days}",
            "update_days": days,
        }
        _ui_print(f"Database will be updated every {days} days.")

    # Configure extraction settings
    _ui_section("Content Extraction Settings")
    _ui_print("Set a page cap for PDF extraction to balance speed vs. coverage.")

    default_pdf_max = existing_semantic_config.get("extraction", {}).get(
        "pdf_max_pages", 10
    )
    if not isinstance(default_pdf_max, int) or default_pdf_max <= 0:
        default_pdf_max = 10

    pdf_max_pages = _ui_int_prompt("PDF max pages", default=default_pdf_max)

    # Configure Zotero database path
    _ui_section("Zotero Database Path")
    _ui_print("By default, zotero-mcp auto-detects the Zotero database location.")
    _ui_print(
        "If Zotero is installed in a custom location, you can specify the path here."
    )

    default_db_path = existing_semantic_config.get("zotero_db_path", "")
    if default_db_path:
        raw_db_path = _ui_prompt("Zotero database path", default=default_db_path)
    else:
        raw_db_path = _ui_prompt(
            "Zotero database path (leave blank for auto-detect)",
            default="",
            show_default=False,
        )

    # Validate path if provided
    zotero_db_path = None
    if raw_db_path:
        db_file = Path(raw_db_path)
        if db_file.exists() and db_file.is_file():
            zotero_db_path = str(db_file)
            _ui_print(f"Using custom Zotero database: {zotero_db_path}")
        else:
            _ui_print(
                f"Warning: File not found at '{raw_db_path}'. Using auto-detect instead."
            )
    elif default_db_path:
        # Keep existing custom path if user just pressed Enter
        zotero_db_path = default_db_path
        _ui_print(f"Keeping existing database path: {zotero_db_path}")
    else:
        _ui_print("Using auto-detect for Zotero database location.")

    config["update_config"] = update_config
    config["extraction"] = {"pdf_max_pages": pdf_max_pages}
    if zotero_db_path:
        config["zotero_db_path"] = zotero_db_path

    return config


def save_semantic_search_config(config: dict, semantic_config_path: Path) -> bool:
    """Save semantic search configuration to file."""
    try:
        # Ensure config directory exists
        semantic_config_dir = semantic_config_path.parent
        semantic_config_dir.mkdir(parents=True, exist_ok=True)

        # Load existing config or create new one
        full_semantic_config = {}
        if semantic_config_path.exists():
            try:
                with open(semantic_config_path) as f:
                    full_semantic_config = json.load(f)
            except json.JSONDecodeError:
                print(
                    "Warning: Existing semantic search config file is invalid JSON, creating new one"
                )

        # Add semantic search config
        full_semantic_config["semantic_search"] = config

        # Write config
        with open(semantic_config_path, "w") as f:
            json.dump(full_semantic_config, f, indent=2)

        print(f"Semantic search configuration saved to: {semantic_config_path}")
        return True

    except Exception as e:
        print(f"Error saving semantic search config: {e}")
        return False


def load_semantic_search_config(semantic_config_path: Path) -> dict:
    """Load existing semantic search configuration."""
    if not semantic_config_path.exists():
        return {}

    try:
        with open(semantic_config_path) as f:
            full_semantic_config = json.load(f)
        return full_semantic_config.get("semantic_search", {})
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse config file as JSON: {e}")
        return {}
    except Exception as e:
        print(f"Warning: Could not read config file: {e}")
        return {}


def update_claude_config(
    config_path,
    zotero_mcp_path,
    local=True,
    api_key=None,
    library_id=None,
    library_type="user",
    semantic_config=None,
):
    """Update Claude Desktop config to add zotero-mcp."""
    # Create directory if it doesn't exist
    config_dir = config_path.parent
    config_dir.mkdir(parents=True, exist_ok=True)

    # Load existing config or create new one
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            print(f"Loaded existing config from: {config_path}")
        except json.JSONDecodeError:
            print(
                f"Error: Config file at {config_path} is not valid JSON. Creating new config."
            )
            config = {}
    else:
        print(f"Creating new config file at: {config_path}")
        config = {}

    # Ensure mcpServers key exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Create environment settings based on local vs web API
    env_settings = {"ZOTERO_LOCAL": "true" if local else "false"}

    # Add API key and library settings for web API
    if not local:
        if api_key:
            env_settings["ZOTERO_API_KEY"] = api_key
        if library_id:
            env_settings["ZOTERO_LIBRARY_ID"] = library_id
        if library_type:
            env_settings["ZOTERO_LIBRARY_TYPE"] = library_type

    # Add semantic search settings if provided
    if semantic_config:
        env_settings["ZOTERO_EMBEDDING_MODEL"] = semantic_config.get(
            "embedding_model", "default"
        )

        embedding_config = semantic_config.get("embedding_config", {})
        if semantic_config.get("embedding_model") == "openai":
            if api_key := embedding_config.get("api_key"):
                env_settings["OPENAI_API_KEY"] = api_key
            if model := embedding_config.get("model_name"):
                env_settings["OPENAI_EMBEDDING_MODEL"] = model
            if base_url := embedding_config.get("base_url"):
                env_settings["OPENAI_BASE_URL"] = base_url

        elif semantic_config.get("embedding_model") == "gemini":
            if api_key := embedding_config.get("api_key"):
                env_settings["GEMINI_API_KEY"] = api_key
            if model := embedding_config.get("model_name"):
                env_settings["GEMINI_EMBEDDING_MODEL"] = model
            if base_url := embedding_config.get("base_url"):
                env_settings["GEMINI_BASE_URL"] = base_url

    # Add or update zotero config
    config["mcpServers"]["zotero"] = {"command": zotero_mcp_path, "env": env_settings}

    # Write updated config
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\nSuccessfully wrote config to: {config_path}")
    except Exception as e:
        print(f"Error writing config file: {str(e)}")
        return False

    return config_path


def _write_standalone_config(
    local: bool,
    api_key: str | None,
    library_id: str | None,
    library_type: str,
    semantic_config: dict,
    no_claude: bool = False,
) -> Path:
    """Write a central config file used by semantic search and provide client env."""
    cfg_dir = Path.home() / ".config" / "zotero-mcp"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "config.json"

    # Load or initialize
    full = {}
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                full = json.load(f)
        except Exception:
            full = {}

    # Store semantic config if provided
    if semantic_config:
        full["semantic_search"] = semantic_config

    # Provide a helper env section for web-based clients
    client_env = {"ZOTERO_LOCAL": "true" if local else "false"}
    # Persist global guard to disable Claude detection/output if requested
    if no_claude:
        client_env["ZOTERO_NO_CLAUDE"] = "true"
    if not local:
        if api_key:
            client_env["ZOTERO_API_KEY"] = api_key
        if library_id:
            client_env["ZOTERO_LIBRARY_ID"] = library_id
        if library_type:
            client_env["ZOTERO_LIBRARY_TYPE"] = library_type

    full["client_env"] = client_env

    with open(cfg_path, "w") as f:
        json.dump(full, f, indent=2)

    return cfg_path


def run_setup(
    no_local: bool = False,
    no_claude: bool = False,
    api_key: str | None = None,
    library_id: str | None = None,
    library_type: Literal["user", "group"] = "user",
    config_path: str | None = None,
    skip_semantic_search: bool = False,
    semantic_config_only: bool = False,
    embedding_model: EmbeddingModelOption | None = None,
    embedding_model_name: str | None = None,
) -> int:
    """Run setup helper logic and return an exit code."""
    # Determine config path for semantic search
    semantic_config_dir = Path.home() / ".config" / "zotero-mcp"
    semantic_config_path = semantic_config_dir / "config.json"
    existing_semantic_config = load_semantic_search_config(semantic_config_path)
    semantic_config_changed = False

    # Handle semantic search only configuration
    if semantic_config_only:
        print("Configuring semantic search only...")
        new_semantic_config = setup_semantic_search(
            existing_semantic_config,
            embedding_model=embedding_model,
            embedding_model_name=embedding_model_name,
        )
        semantic_config_changed = existing_semantic_config != new_semantic_config
        # only save if semantic config changed
        if semantic_config_changed:
            if save_semantic_search_config(new_semantic_config, semantic_config_path):
                print("\nSemantic search configuration complete!")
                print(f"Configuration saved to: {semantic_config_path}")
                print("\nTo initialize the database, run: zotero-mcp update-db")
                return 0
            print("\nSemantic search configuration failed.")
            return 1

        print("\nSemantic search configuration left unchanged.")
        return 0

    # Find zotero-mcp executable
    exe_path = find_executable()
    if not exe_path:
        print("Error: Could not find zotero-mcp executable.")
        return 1
    print(f"Using zotero-mcp at: {exe_path}")

    # Find Claude Desktop config unless --no-claude
    resolved_config_path: Path | None = None
    if not no_claude:
        if config_path:
            print(f"Using specified config path: {config_path}")
            resolved_config_path = Path(config_path)
        else:
            resolved_config_path = find_claude_config()

        if not resolved_config_path:
            print("Error: Could not determine Claude Desktop config path.")
            return 1

    # Update config
    use_local = not no_local

    # Configure semantic search if not skipped
    if not skip_semantic_search:
        should_configure_semantic = embedding_model is not None

        if should_configure_semantic:
            print("\nApplying semantic search embedding model from CLI options...")
        else:
            # if there is already a semantic search configuration in the config file:
            if existing_semantic_config:
                print(
                    "\nFound an exisiting semantic search configuration in the config file."
                )
                print("Would you like to reconfigure semantic search? (y/n): ", end="")
            # if otherwise, slightly different message...
            else:
                print("\nWould you like to configure semantic search? (y/n): ", end="")
            should_configure_semantic = input().strip().lower() in ["y", "yes"]

        if should_configure_semantic:
            new_semantic_config = setup_semantic_search(
                existing_semantic_config,
                embedding_model=embedding_model,
                embedding_model_name=embedding_model_name,
            )
            if existing_semantic_config != new_semantic_config:
                semantic_config_changed = True
                existing_semantic_config = new_semantic_config  # Update config in use
                save_semantic_search_config(
                    existing_semantic_config, semantic_config_path
                )
    elif embedding_model is not None or embedding_model_name is not None:
        print(
            "Warning: --embedding-model/--embedding-model-name were ignored because --skip-semantic-search is set."
        )

    print("\nSetup with the following settings:")
    print(f"  Local API: {use_local}")
    if not use_local:
        print(f"  API Key: {api_key or 'Not provided'}")
        print(f"  Library ID: {library_id or 'Not provided'}")
        print(f"  Library Type: {library_type}")

    # Use the potentially updated semantic config
    semantic_config = existing_semantic_config

    # Update configuration based on mode
    try:
        if no_claude:
            cfg_path = _write_standalone_config(
                local=use_local,
                api_key=api_key,
                library_id=library_id,
                library_type=library_type,
                semantic_config=semantic_config,
                no_claude=no_claude,
            )
            print("\nSetup complete (standalone/web mode)!")
            print(f"Config saved to: {cfg_path}")
            # Emit one-line client_env for easy copy/paste
            try:
                with open(cfg_path) as f:
                    full = json.load(f)
                env_line = json.dumps(full.get("client_env", {}), separators=(",", ":"))
                print("Client environment (single-line JSON):")
                print(env_line)
            except Exception:
                pass
            if semantic_config_changed:
                print(
                    "\nNote: You changed semantic search settings. Consider rebuilding the DB:"
                )
                print("  zotero-mcp update-db --force-rebuild")
            return 0

        assert resolved_config_path is not None
        updated_config_path = update_claude_config(
            resolved_config_path,
            exe_path,
            local=use_local,
            api_key=api_key,
            library_id=library_id,
            library_type=library_type,
            semantic_config=semantic_config,
        )
        if updated_config_path:
            print("\nSetup complete!")
            print("To use Zotero in Claude Desktop:")
            print("1. Restart Claude Desktop if it's running")
            print("2. In Claude, type: /tools zotero")
            if semantic_config_changed:
                print("\nSemantic Search:")
                print(
                    "- Configured with",
                    semantic_config.get("embedding_model", "default"),
                    "embedding model",
                )
                print(
                    "- To change the configuration, run: zotero-mcp setup --semantic-config-only"
                )
                print(
                    "- The config file is located at: ~/.config/zotero-mcp/config.json"
                )
                print(
                    "- You may need to rebuild your database: zotero-mcp update-db --force-rebuild"
                )
            else:
                print("\nSemantic Search:")
                print("- To update the database, run: zotero-mcp update-db")
                print(
                    "- Use zotero_semantic_search tool in Claude for AI-powered search"
                )
            if use_local:
                print(
                    "\nNote: Make sure Zotero desktop is running and the local API is enabled in preferences."
                )
            else:
                missing = []
                if not api_key:
                    missing.append("API key")
                if not library_id:
                    missing.append("Library ID")
                if missing:
                    print(
                        f"\nWarning: The following required settings for Web API were not provided: {', '.join(missing)}"
                    )
                    print(
                        "You may need to set these as environment variables or reconfigure."
                    )
            return 0

        print("\nSetup failed. See errors above.")
        return 1
    except Exception as e:
        print(f"\nSetup failed with error: {str(e)}")
        return 1


def main(
    no_local: Annotated[
        bool,
        typer.Option(
            "--no-local", help="Configure for Zotero Web API instead of local API"
        ),
    ] = False,
    no_claude: Annotated[
        bool,
        typer.Option(
            "--no-claude",
            help="Don't setup Claude Desktop config: instead store settings in config file.",
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
        EmbeddingModelOption | None,
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
) -> None:
    """Configure zotero-mcp for Claude Desktop or standalone use."""
    exit_code = run_setup(
        no_local=no_local,
        no_claude=no_claude,
        api_key=api_key,
        library_id=library_id,
        library_type=library_type,
        config_path=config_path,
        skip_semantic_search=skip_semantic_search,
        semantic_config_only=semantic_config_only,
        embedding_model=embedding_model,
        embedding_model_name=embedding_model_name,
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


if __name__ == "__main__":
    typer.run(main)
