# LLAMA.rs Chat Example

This is a simple CLI chat application that uses GGUF models with the llama-cpp-2 Rust bindings. It allows you to have interactive conversations with various LLM models in GGUF format.

## Features

- Load local GGUF models or download them from Hugging Face
- Interactive chat interface with command history
- Configurable system prompt
- GPU acceleration support (with CUDA or Vulkan features enabled)
- Adjustable context size and other model parameters

## Building

```bash
# Build with default features
cargo build --release

# Build with CUDA support
cargo build --release --features cuda

# Build with Vulkan support
cargo build --release --features vulkan
```

## Usage

### Using a local model:

```bash
./target/release/llama-rs-example local /path/to/your/model.gguf
```

### Using a model from Hugging Face:

```bash
./target/release/llama-rs-example hf-model "TheBloke/Llama-2-7B-Chat-GGUF" "llama-2-7b-chat.Q4_K_M.gguf"
```

### Additional Options

- `--system-prompt` - Set a custom system prompt (default: "You are a helpful AI assistant...")
- `--ctx-size` - Set the context window size in tokens (default: 2048)
- `-t, --threads` - Number of threads to use for generation
- `-v, --verbose` - Enable verbose logging
- `--disable-gpu` - Disable GPU acceleration (when built with CUDA/Vulkan support)

## Example

```bash
# Run with a local model and custom system prompt
./target/release/llama-rs-example local ./models/mistral-7b.gguf --system-prompt "You are a helpful coding assistant."
```

Once running, you can type your messages and press Enter to chat. Type 'exit' to quit the program. 
