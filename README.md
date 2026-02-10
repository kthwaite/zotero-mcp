# Zotero MCP: Chat with your Research Library—Local or Web—in Claude, ChatGPT, and more.

<p align="center">
  <a href="https://www.zotero.org/">
    <img src="https://img.shields.io/badge/Zotero-CC2936?style=for-the-badge&logo=zotero&logoColor=white" alt="Zotero">
  </a>
  <a href="https://www.anthropic.com/claude">
    <img src="https://img.shields.io/badge/Claude-6849C3?style=for-the-badge&logo=anthropic&logoColor=white" alt="Claude">
  </a>
  <a href="https://chatgpt.com/">
    <img src="https://img.shields.io/badge/ChatGPT-74AA9C?style=for-the-badge&logo=openai&logoColor=white" alt="ChatGPT">
  </a>
  <a href="https://modelcontextprotocol.io/introduction">
    <img src="https://img.shields.io/badge/MCP-0175C2?style=for-the-badge&logoColor=white" alt="MCP">
  </a>
</p>

**Zotero MCP** seamlessly connects your [Zotero](https://www.zotero.org/) research library with [ChatGPT](https://openai.com), [Claude](https://www.anthropic.com/claude), and other AI assistants (e.g., [Cherry Studio](https://cherry-ai.com/), [Chorus](https://chorus.sh), [Cursor](https://www.cursor.com/)) via the [Model Context Protocol](https://modelcontextprotocol.io/introduction). Review papers, get summaries, analyze citations, extract PDF annotations, and more!

## ✨ Features

### 🧠 AI-Powered Semantic Search
- **Vector-based similarity search** over your entire research library
- **Multiple embedding models**: Default (free), OpenAI, and Gemini options
- **Intelligent results** with similarity scores and contextual matching
- **Auto-updating database** with configurable sync schedules

### 🔍 Search Your Library
- Find papers, articles, and books by title, author, or content
- Perform complex searches with multiple criteria
- Browse collections, tags, and recent additions
- **NEW**: Semantic search for conceptual and topic-based discovery

### 📚 Access Your Content
- Retrieve detailed metadata for any item
- Get full text content (when available)
- Access attachments, notes, and child items

### 📝 Work with Annotations
- Extract and search PDF annotations directly
- Access Zotero's native annotations
- Create and update notes and annotations

### 🔄 Easy Updates
- **Smart update system** that detects your installation method (uv, pip, conda, pipx)
- **Configuration preservation** - all settings maintained during updates
- **Version checking** and automatic update notifications

### 🌐 Flexible Access Methods
- Local method for offline access (no API key needed)
- Web API for cloud library access
- Perfect for both local research and remote collaboration

## 🚀 Quick Install

### Default Installation

#### Installing via uv

```bash
uv tool install "git+https://github.com/54yyyu/zotero-mcp.git"
zotero-mcp config init  # Auto-configure (Claude Desktop supported)
```

#### Installing via pip

```bash
pip install git+https://github.com/54yyyu/zotero-mcp.git
zotero-mcp config init  # Auto-configure (Claude Desktop supported)
```

### Installing via Smithery

To install Zotero MCP via [Smithery](https://smithery.ai/server/@54yyyu/zotero-mcp) for Claude Desktop:

```bash
npx -y @smithery/cli install @54yyyu/zotero-mcp --client claude
```

#### Updating Your Installation

Keep zotero-mcp up to date with the smart update command:

```bash
# Check for updates
zotero-mcp self update --check-only

# Update to latest version (preserves all configurations)
zotero-mcp self update
```

## 🧠 Semantic Search

Zotero MCP now includes powerful AI-powered semantic search capabilities that let you find research based on concepts and meaning, not just keywords.

### Setup Semantic Search

During setup or separately, configure semantic search:

```bash
# Configure during initial setup (recommended)
zotero-mcp config init

# Launch full-screen setup wizard (Textual TUI)
zotero-mcp config init --tui

# Or configure semantic search separately
zotero-mcp config init --semantic-config-only
```

**Available Embedding Models:**
- **MiniLM (default / `minilm`)**: Free, runs locally, good for most use cases (`all-MiniLM-L6-v2`)
- **Qwen (`qwen`)**: Local HuggingFace model (`Qwen/Qwen3-Embedding-0.6B`)
- **EmbeddingGemma (`embeddinggemma`)**: Local HuggingFace model (`google/embeddinggemma-300m`)
- **Custom HuggingFace (`custom-hf`)**: Any local/remote HF model ID via `--embedding-model`
- **OpenAI (`openai`)**: API-based (`text-embedding-3-small` or `text-embedding-3-large`)
- **Gemini (`gemini`)**: API-based (`models/text-embedding-004` or experimental models)

You can preselect models from the CLI:

```bash
# Non-default local model
zotero-mcp config init --semantic-config-only --embedding-backend qwen

# Custom HuggingFace model
zotero-mcp config init --semantic-config-only --embedding-backend custom-hf \
  --embedding-model sentence-transformers/all-mpnet-base-v2

# Optional: pre-download local embedding weights now to avoid first-run lag
zotero-mcp embeddings warmup
```

**Update Frequency Options:**
- **Manual**: Update only when you run `zotero-mcp index sync`
- **Auto on startup**: Update database every time the server starts
- **Daily**: Update once per day automatically
- **Every N days**: Set custom interval

### Using Semantic Search

After setup, initialize your search database:

```bash
# Build the semantic search database (fast, metadata-only)
zotero-mcp index sync

# Build with full-text extraction (slower, more comprehensive)
zotero-mcp index sync --fulltext

# Use your custom zotero.sqlite path
zotero-mcp index sync --fulltext --db-path "/Your_custom_path/zotero.sqlite"

# Check database status
zotero-mcp index status
```

If your configured embedding model changes (for example MiniLM → Qwen), Zotero MCP now detects the mismatch and resets the Chroma collection by default so new embeddings are created with the selected model. You can still use `--force-rebuild` to proactively rebuild everything.

**Example Semantic Queries in your AI assistant:**
- *"Find research similar to machine learning concepts in neuroscience"*
- *"Papers that discuss climate change impacts on agriculture"*
- *"Research related to quantum computing applications"*
- *"Studies about social media influence on mental health"*
- *"Find papers conceptually similar to this abstract: [paste abstract]"*

The semantic search provides similarity scores and finds papers based on conceptual understanding, not just keyword matching.

## 🖥️ Setup & Usage

Full documentation is available at [Zotero MCP docs](https://stevenyuyy.us/zotero-mcp/).

**Requirements**
- Python 3.10+
- Zotero 7+ (for local API with full-text access)
- An MCP-compatible client (e.g., Claude Desktop, ChatGPT Developer Mode, Cherry Studio, Chorus)

**For ChatGPT setup: see the [Getting Started guide](./docs/getting-started.md).**

### For Claude Desktop (example MCP client)

#### Configuration
After installation, either:

1. **Auto-configure** (recommended):
   ```bash
   zotero-mcp config init
   ```

2. **Manual configuration**:
   Add to your `claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "zotero": {
         "command": "zotero-mcp",
         "env": {
           "ZOTERO_LOCAL": "true"
         }
       }
     }
   }
   ```

#### Usage

1. Start Zotero desktop (make sure local API is enabled in preferences)
2. Launch Claude Desktop
3. Access the Zotero-MCP tool through Claude Desktop's tools interface

Example prompts:
- "Search my library for papers on machine learning"
- "Find recent articles I've added about climate change"
- "Summarize the key findings from my paper on quantum computing"
- "Extract all PDF annotations from my paper on neural networks"
- "Search my notes and annotations for mentions of 'reinforcement learning'"
- "Show me papers tagged '#Arm' excluding those with '#Crypt' in my library"
- "Search for papers on operating system with tag '#Arm'"
- "Export the BibTeX citation for papers on machine learning"
- **"Find papers conceptually similar to deep learning in computer vision"** *(semantic search)*
- **"Research that relates to the intersection of AI and healthcare"** *(semantic search)*
- **"Papers that discuss topics similar to this abstract: [paste text]"** *(semantic search)*

### For Cherry Studio

#### Configuration
Go to Settings -> MCP Servers -> Edit MCP Configuration, and add the following:

```json
{
  "mcpServers": {
    "zotero": {
      "name": "zotero",
      "type": "stdio",
      "isActive": true,
      "command": "zotero-mcp",
      "args": [],
      "env": {
        "ZOTERO_LOCAL": "true"
      }
    }
  }
}
```
Then click "Save".

Cherry Studio also provides a visual configuration method for general settings and tools selection.

## 🔧 Advanced Configuration

### Using Web API Instead of Local API

For accessing your Zotero library via the web API (useful for remote setups):

```bash
zotero-mcp config init --mode web --api-key YOUR_API_KEY --library-id YOUR_LIBRARY_ID
```

### Environment Variables

**Zotero Connection:**
- `ZOTERO_LOCAL=true`: Use the local Zotero API (default: false)
- `ZOTERO_API_KEY`: Your Zotero API key (for web API)
- `ZOTERO_LIBRARY_ID`: Your Zotero library ID (for web API)
- `ZOTERO_LIBRARY_TYPE`: The type of library (user or group, default: user)

**Semantic Search:**
- `ZOTERO_EMBEDDING_MODEL`: Embedding model to use (default, qwen, embeddinggemma, openai, gemini, or custom HF model ID)
- `ZOTERO_EMBEDDING_MISMATCH_BEHAVIOR`: What to do if an existing collection was built with a different embedding (`reset` (default), `reuse`, `error`)
- `OPENAI_API_KEY`: Your OpenAI API key (for OpenAI embeddings)
- `OPENAI_EMBEDDING_MODEL`: OpenAI model name (text-embedding-3-small, text-embedding-3-large)
- `OPENAI_BASE_URL`: Custom OpenAI endpoint URL (optional, for use with compatible APIs)
- `GEMINI_API_KEY`: Your Gemini API key (for Gemini embeddings)
- `GEMINI_EMBEDDING_MODEL`: Gemini model name (models/text-embedding-004, etc.)
- `GEMINI_BASE_URL`: Custom Gemini endpoint URL (optional, for use with compatible APIs)
- `ZOTERO_DB_PATH`: Custom `zotero.sqlite` path (optional)

### Command-Line Options

```bash
# Run the server directly
zotero-mcp server start

# Specify transport method
zotero-mcp server start --transport stdio|streamable-http|sse

# Setup and configuration
zotero-mcp config init --help                                  # Get help on setup options
zotero-mcp config init --tui                                   # Launch full-screen Textual setup wizard
zotero-mcp config init --semantic-config-only                  # Configure only semantic search
zotero-mcp config init --semantic-config-only --embedding-backend minilm|qwen|embeddinggemma|custom-hf|openai|gemini
zotero-mcp config init --semantic-config-only --embedding-model MODEL_NAME
zotero-mcp config show                                         # Show installation path and config info for MCP clients

# Updates and maintenance
zotero-mcp self update                         # Update to latest version
zotero-mcp self update --check-only            # Check for updates without installing
zotero-mcp self update --force                 # Force update even if up to date

# Semantic search database management
zotero-mcp embeddings warmup                   # Pre-download/warm local embedding model weights
zotero-mcp index sync                          # Update semantic search database (fast, metadata-only)
zotero-mcp index sync --fulltext               # Update with full-text extraction (comprehensive but slower)
zotero-mcp index sync --force-rebuild          # Force complete database rebuild
zotero-mcp index sync --fulltext --force-rebuild  # Rebuild with full-text extraction
zotero-mcp index sync --fulltext --db-path "your_path_to/zotero.sqlite" # Customize your zotero database path
zotero-mcp index status                        # Show database status and info
zotero-mcp index inspect                       # Inspect indexed records / stats

# General
zotero-mcp self version                        # Show current version
zotero-mcp version                             # Top-level alias
```

> Legacy top-level commands (`setup`, `serve`, `update-db`, etc.) still work for now and print deprecation warnings.

## 📑 PDF Annotation Extraction

Zotero MCP includes advanced PDF annotation extraction capabilities:

- **Direct PDF Processing**: Extract annotations directly from PDF files, even if they're not yet indexed by Zotero
- **Enhanced Search**: Search through PDF annotations and comments
- **Image Annotation Support**: Extract image annotations from PDFs
- **Seamless Integration**: Works alongside Zotero's native annotation system

For optimal annotation extraction, it is **highly recommended** to install the [Better BibTeX plugin](https://retorque.re/zotero-better-bibtex/installation/) for Zotero. The annotation-related functions have been primarily tested with this plugin and provide enhanced functionality when it's available.


The first time you use PDF annotation features, the necessary tools will be automatically downloaded.

## 📚 Available Tools

### 🧠 Semantic Search Tools
- `zotero_semantic_search`: AI-powered similarity search with embedding models
- `zotero_update_search_database`: Manually update the semantic search database
- `zotero_get_search_database_status`: Check database status and configuration

### 🔍 Search Tools
- `zotero_search_items`: Search your library by keywords
- `zotero_advanced_search`: Perform complex searches with multiple criteria
- `zotero_get_collections`: List collections
- `zotero_get_collection_items`: Get items in a collection
- `zotero_get_tags`: List all tags
- `zotero_get_recent`: Get recently added items
- `zotero_search_by_tag`: Search your library using custom tag filters

### 📚 Content Tools
- `zotero_get_item_metadata`: Get detailed metadata (supports BibTeX export via `format="bibtex"`)
- `zotero_get_item_fulltext`: Get full text content
- `zotero_get_item_children`: Get attachments and notes

### 📝 Annotation & Notes Tools
- `zotero_get_annotations`: Get annotations (including direct PDF extraction)
- `zotero_get_notes`: Retrieve notes from your Zotero library
- `zotero_search_notes`: Search in notes and annotations (including PDF-extracted)
- `zotero_create_note`: Create a new note for an item (beta feature)

## 🔍 Troubleshooting

### General Issues
- **No results found**: Ensure Zotero is running and the local API is enabled. You need to toggle on `Allow other applications on this computer to communicate with Zotero` in Zotero preferences.
- **Can't connect to library**: Check your API key and library ID if using web API
- **Full text not available**: Make sure you're using Zotero 7+ for local full-text access
- **Local library limitations**: Some functionality (tagging, library modifications) may not work with local JS API. Consider using web library setup for full functionality. (See the [docs](docs/getting-started.md#local-library-limitations) for more info.)
- **Installation/search option switching issues**: Database problems from changing install methods or search options can often be resolved with `zotero-mcp index sync --force-rebuild`

### Semantic Search Issues
- **"Missing required environment variables" when running index sync**: Run `zotero-mcp config init` to configure your environment, or the CLI will automatically load settings from your MCP client config (e.g., Claude Desktop)
- **ChromaDB warnings**: Update to the latest version - deprecation warnings have been fixed
- **Database update takes long**: By default, `index sync` is fast (metadata-only). For comprehensive indexing with full-text, use `--fulltext` flag. Use `--limit` parameter for testing: `zotero-mcp index sync --limit 100`
- **Server startup is slow with local embedding models**: Pre-download model weights once with `zotero-mcp embeddings warmup`
- **Semantic search returns no results**: Ensure the database is initialized with `zotero-mcp index sync` and check status with `zotero-mcp index status`
- **Limited search quality**: For better semantic search results, use `zotero-mcp index sync --fulltext` to index full-text content (requires local Zotero setup)
- **OpenAI/Gemini API errors**: Verify your API keys are correctly set and have sufficient credits/quota

### Update Issues
- **Update command fails**: Check your internet connection and try `zotero-mcp self update --force`
- **Configuration lost after update**: The update process preserves configs automatically, but check `~/.config/zotero-mcp/` for backup files

## 📄 License

MIT
