"""Textual-based setup wizard for zotero-mcp."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
)

EmbeddingModelOption = Literal[
    "minilm", "qwen", "embeddinggemma", "custom-hf", "openai", "gemini"
]
LibraryType = Literal["user", "group"]

_KNOWN_EMBEDDING_MODELS = {
    "default",
    "qwen",
    "embeddinggemma",
    "openai",
    "gemini",
}


@dataclass(slots=True)
class SetupWizardDefaults:
    no_local: bool = False
    no_claude: bool = False
    api_key: str | None = None
    library_id: str | None = None
    library_type: LibraryType = "user"
    embedding_model: EmbeddingModelOption | None = None
    embedding_model_name: str | None = None
    skip_semantic_search: bool = False


@dataclass(slots=True)
class SetupWizardResult:
    no_local: bool
    no_claude: bool
    api_key: str | None
    library_id: str | None
    library_type: LibraryType
    configure_semantic: bool
    semantic_config: dict | None


class SetupWizardApp(App[SetupWizardResult | None]):
    """Full-screen setup wizard using Textual widgets."""

    CSS = """
    Screen {
        align: center middle;
    }

    #wizard {
        width: 92%;
        height: 92%;
        border: round $accent;
        padding: 1 2;
    }

    #title {
        content-align: center middle;
        text-style: bold;
        margin-bottom: 1;
    }

    .section {
        text-style: bold;
        margin-top: 1;
    }

    .field-label {
        margin-top: 1;
    }

    #status {
        color: $error;
        height: 2;
        margin-top: 1;
    }

    #actions {
        margin-top: 1;
        align-horizontal: right;
        height: 3;
    }

    #actions Button {
        margin-left: 1;
    }
    """

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(
        self,
        *,
        existing_semantic_config: dict | None,
        defaults: SetupWizardDefaults,
        semantic_config_only: bool,
    ):
        super().__init__()
        self.existing_semantic_config = existing_semantic_config or {}
        self.defaults = defaults
        self.semantic_config_only = semantic_config_only

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with VerticalScroll(id="wizard"):
            yield Static("Zotero MCP Setup Wizard", id="title")

            if not self.semantic_config_only:
                yield Label("Zotero connection", classes="section")
                yield Checkbox(
                    "Use local Zotero API",
                    value=not self.defaults.no_local,
                    id="use_local",
                )
                yield Label("Zotero API key (for Web API)", classes="field-label")
                yield Input(
                    value=self.defaults.api_key or "",
                    placeholder="Optional unless using Web API",
                    password=True,
                    id="api_key",
                )
                yield Label("Zotero library ID (for Web API)", classes="field-label")
                yield Input(
                    value=self.defaults.library_id or "",
                    placeholder="Optional unless using Web API",
                    id="library_id",
                )
                yield Label("Zotero library type", classes="field-label")
                yield Select[str](
                    (("User", "user"), ("Group", "group")),
                    value=self.defaults.library_type,
                    id="library_type",
                )
                yield Checkbox(
                    "Configure Claude Desktop integration",
                    value=not self.defaults.no_claude,
                    id="configure_claude",
                )

            yield Label("Semantic search", classes="section")
            yield Checkbox(
                "Configure semantic search",
                value=not self.defaults.skip_semantic_search,
                id="configure_semantic",
            )

            yield Label("Embedding backend", classes="field-label")
            yield Select[str](
                (
                    ("MiniLM (default)", "minilm"),
                    ("Qwen", "qwen"),
                    ("EmbeddingGemma", "embeddinggemma"),
                    ("Custom HuggingFace", "custom-hf"),
                    ("OpenAI", "openai"),
                    ("Gemini", "gemini"),
                ),
                value=self._default_embedding_choice(),
                id="embedding_model",
            )

            yield Label(
                "Model name override / custom HF model ID", classes="field-label"
            )
            yield Input(
                value=self._default_embedding_model_name(),
                placeholder="Optional (required for custom-hf)",
                id="embedding_model_name",
            )

            yield Label(
                "Embedding provider API key (OpenAI/Gemini)", classes="field-label"
            )
            yield Input(
                value=self._existing_embedding_api_key(),
                password=True,
                placeholder="Optional unless using OpenAI/Gemini",
                id="embedding_api_key",
            )

            yield Label("Embedding provider base URL", classes="field-label")
            yield Input(
                value=self._existing_embedding_base_url(),
                placeholder="Optional custom endpoint",
                id="embedding_base_url",
            )

            yield Label("Semantic DB update frequency", classes="field-label")
            yield Select[str](
                (
                    ("Manual", "manual"),
                    ("On startup", "startup"),
                    ("Daily", "daily"),
                    ("Every N days", "every_n"),
                ),
                value=self._default_update_choice(),
                id="update_frequency",
            )

            yield Label(
                "Days between updates (for Every N days)", classes="field-label"
            )
            yield Input(value=str(self._default_update_days()), id="update_days")

            yield Label("PDF max pages for extraction", classes="field-label")
            yield Input(value=str(self._default_pdf_max_pages()), id="pdf_max_pages")

            yield Label("Zotero database path (optional)", classes="field-label")
            yield Input(value=self._default_db_path(), id="zotero_db_path")

            yield Static("", id="status")

            with Horizontal(id="actions"):
                yield Button("Cancel", id="cancel")
                yield Button("Apply", id="apply", variant="primary")

        yield Footer()

    def action_cancel(self) -> None:
        self.exit(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.exit(None)
            return

        if event.button.id == "apply":
            result = self._build_result()
            if result is not None:
                self.exit(result)

    def _default_embedding_choice(self) -> EmbeddingModelOption:
        if self.defaults.embedding_model:
            return self.defaults.embedding_model

        existing_model = self.existing_semantic_config.get("embedding_model", "default")
        if existing_model == "default":
            return "minilm"
        if existing_model in {"qwen", "embeddinggemma", "openai", "gemini"}:
            return cast(EmbeddingModelOption, existing_model)
        return "custom-hf"

    def _default_embedding_model_name(self) -> str:
        if self.defaults.embedding_model_name:
            return self.defaults.embedding_model_name

        existing_model = self.existing_semantic_config.get("embedding_model", "")
        embedding_config = self.existing_semantic_config.get("embedding_config", {})
        configured_name = embedding_config.get("model_name", "")

        if configured_name:
            return str(configured_name)
        if existing_model and existing_model not in _KNOWN_EMBEDDING_MODELS:
            return str(existing_model)
        return ""

    def _existing_embedding_api_key(self) -> str:
        embedding_config = self.existing_semantic_config.get("embedding_config", {})
        return str(embedding_config.get("api_key", ""))

    def _existing_embedding_base_url(self) -> str:
        embedding_config = self.existing_semantic_config.get("embedding_config", {})
        return str(embedding_config.get("base_url", ""))

    def _default_update_choice(self) -> str:
        update_frequency = self.existing_semantic_config.get("update_config", {}).get(
            "update_frequency", "manual"
        )
        if update_frequency in {"manual", "startup", "daily"}:
            return str(update_frequency)
        if str(update_frequency).startswith("every_"):
            return "every_n"
        return "manual"

    def _default_update_days(self) -> int:
        update_config = self.existing_semantic_config.get("update_config", {})
        days = update_config.get("update_days")
        if isinstance(days, int) and days > 0:
            return days

        update_frequency = str(update_config.get("update_frequency", ""))
        if update_frequency.startswith("every_"):
            try:
                parsed = int(update_frequency.split("_", 1)[1])
                if parsed > 0:
                    return parsed
            except Exception:
                pass

        return 7

    def _default_pdf_max_pages(self) -> int:
        value = self.existing_semantic_config.get("extraction", {}).get(
            "pdf_max_pages", 10
        )
        if isinstance(value, int) and value > 0:
            return value
        return 10

    def _default_db_path(self) -> str:
        value = self.existing_semantic_config.get("zotero_db_path", "")
        return str(value) if value else ""

    def _set_error(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    def _select_value(self, widget_id: str, default: str) -> str:
        value = self.query_one(f"#{widget_id}", Select).value
        return value if isinstance(value, str) else default

    @staticmethod
    def _optional_text(value: str) -> str | None:
        text = value.strip()
        return text or None

    def _parse_positive_int(self, widget_id: str, field_name: str) -> int | None:
        raw = self.query_one(f"#{widget_id}", Input).value.strip()
        try:
            value = int(raw)
        except ValueError:
            self._set_error(f"{field_name} must be a number.")
            return None

        if value <= 0:
            self._set_error(f"{field_name} must be greater than zero.")
            return None

        return value

    def _build_semantic_config(self) -> dict | None:
        configure_semantic = self.query_one("#configure_semantic", Checkbox).value
        if not configure_semantic:
            return None

        embedding_choice = self._select_value("embedding_model", "minilm")
        embedding_model_name = self.query_one(
            "#embedding_model_name", Input
        ).value.strip()
        embedding_api_key = self.query_one("#embedding_api_key", Input).value.strip()
        embedding_base_url = self.query_one("#embedding_base_url", Input).value.strip()

        config: dict[str, object] = {}

        if embedding_choice == "minilm":
            config["embedding_model"] = "default"

        elif embedding_choice == "qwen":
            config["embedding_model"] = "qwen"
            if embedding_model_name:
                config["embedding_config"] = {"model_name": embedding_model_name}

        elif embedding_choice == "embeddinggemma":
            config["embedding_model"] = "embeddinggemma"
            if embedding_model_name:
                config["embedding_config"] = {"model_name": embedding_model_name}

        elif embedding_choice == "custom-hf":
            if not embedding_model_name:
                self._set_error(
                    "Custom HuggingFace requires a model ID in 'Model name override / custom HF model ID'."
                )
                return None
            config["embedding_model"] = embedding_model_name

        elif embedding_choice == "openai":
            config["embedding_model"] = "openai"
            model_name = embedding_model_name or "text-embedding-3-small"
            embedding_config: dict[str, str] = {"model_name": model_name}
            if embedding_api_key:
                embedding_config["api_key"] = embedding_api_key
            if embedding_base_url:
                embedding_config["base_url"] = embedding_base_url
            config["embedding_config"] = embedding_config

        elif embedding_choice == "gemini":
            config["embedding_model"] = "gemini"
            model_name = embedding_model_name or "models/text-embedding-004"
            embedding_config = {"model_name": model_name}
            if embedding_api_key:
                embedding_config["api_key"] = embedding_api_key
            if embedding_base_url:
                embedding_config["base_url"] = embedding_base_url
            config["embedding_config"] = embedding_config

        update_frequency = self._select_value("update_frequency", "manual")
        if update_frequency == "manual":
            config["update_config"] = {
                "auto_update": False,
                "update_frequency": "manual",
            }
        elif update_frequency == "startup":
            config["update_config"] = {
                "auto_update": True,
                "update_frequency": "startup",
            }
        elif update_frequency == "daily":
            config["update_config"] = {
                "auto_update": True,
                "update_frequency": "daily",
            }
        else:
            days = self._parse_positive_int("update_days", "Update days")
            if days is None:
                return None
            config["update_config"] = {
                "auto_update": True,
                "update_frequency": f"every_{days}",
                "update_days": days,
            }

        pdf_max_pages = self._parse_positive_int("pdf_max_pages", "PDF max pages")
        if pdf_max_pages is None:
            return None
        config["extraction"] = {"pdf_max_pages": pdf_max_pages}

        raw_db_path = self.query_one("#zotero_db_path", Input).value.strip()
        if raw_db_path:
            db_file = Path(raw_db_path)
            if db_file.exists() and db_file.is_file():
                config["zotero_db_path"] = str(db_file)
            else:
                self._set_error(
                    f"Warning: '{raw_db_path}' was not found; using auto-detect instead."
                )

        return config

    def _build_result(self) -> SetupWizardResult | None:
        self._set_error("")

        if self.semantic_config_only:
            no_local = self.defaults.no_local
            no_claude = self.defaults.no_claude
            api_key = self.defaults.api_key
            library_id = self.defaults.library_id
            library_type = self.defaults.library_type
        else:
            use_local = self.query_one("#use_local", Checkbox).value
            no_local = not use_local
            no_claude = not self.query_one("#configure_claude", Checkbox).value

            api_key = self._optional_text(self.query_one("#api_key", Input).value)
            library_id = self._optional_text(self.query_one("#library_id", Input).value)
            library_type = cast(
                LibraryType,
                self._select_value("library_type", self.defaults.library_type),
            )

        configure_semantic = self.query_one("#configure_semantic", Checkbox).value
        semantic_config = self._build_semantic_config()
        if configure_semantic and semantic_config is None:
            # Validation failed and error is shown to the user.
            return None

        return SetupWizardResult(
            no_local=no_local,
            no_claude=no_claude,
            api_key=api_key,
            library_id=library_id,
            library_type=library_type,
            configure_semantic=configure_semantic,
            semantic_config=semantic_config,
        )


def run_setup_textual_wizard(
    *,
    existing_semantic_config: dict | None,
    semantic_config_only: bool,
    defaults: SetupWizardDefaults,
) -> SetupWizardResult | None:
    """Run the setup Textual app and return wizard choices."""
    app = SetupWizardApp(
        existing_semantic_config=existing_semantic_config,
        defaults=defaults,
        semantic_config_only=semantic_config_only,
    )
    return app.run()
