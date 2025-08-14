# NexusVoice

NexusVoice is a Python-based voice processing application using libraries like `pyaudio`, `openwakeword`, and others. It features audio streaming, wake word detection, and support for real-time inference.

## Project Setup

### 1. Clone the repository

```sh
git clone https://github.com/madsmith/NexusVoice.git
cd NexusVoice
```

### 2. Create and activate a virtual environment

```sh
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install system dependencies

#### macOS

```sh
xcode-select --install     # If not already installed
brew install portaudio     # Required for PyAudio
brew install ffmpeg        # Required for audio processing
```

#### Linux

```sh
sudo apt install portaudio19-dev ffmpeg
```

### 4. Install Python dependencies

NexusVoice uses `uv` for dependency management. If you don't have it installed, [install it first](https://github.com/astral-sh/uv):

```sh
pip install uv
```

#### Full Project Installation (for developers)

Install the entire project in editable mode with development dependencies:

```sh
uv pip install -e ".[dev]"
```

#### Submodule Installation Options

NexusVoice is organized with modular subpackages. You can install just what you need:

**Core Module Only** (minimal dependencies for projects using just NexusConfig):

```sh
uv pip install -e ./nexusvoice/core
```

**Development Workflow** (when working on both core and main project):

```sh
# Install core first, then main project
uv pip install -e ./nexusvoice/core
uv pip install -e .
```

## Usage

Example scripts that test dev functionality are in examples.  The primary entry points are NexusClient

```sh
python NexusClient.py
```

## Project Structure

```none
nexusvoice/         # Core application package  
examples/           # Demo and test scripts  
recordings/         # Saved audio recordings  
models/             # ONNX models (optional, configurable)  
pyproject.toml      # Project metadata and dependencies
```

## Notes

- Ensure `PYTORCH_ENABLE_MPS_FALLBACK=1` is set if you're on a Mac with MPS but need CPU fallback.
- You may suppress fallback warnings using Python warnings filters (already handled in the app).

## Dev Dependencies

For contributors or testers, dev dependencies are installed via the `dev` extra:

```sh
pip install -e .[dev]
```

This includes optional tools like:

- `matplotlib` – for plotting or debugging