# Llama Rust Reference Implementation

This is my reference implementation of a high level wrapper around llama-cpp-rs. The default implementation is quite low level and requires a lot of boilerplate to use, and I wanted a simple interface for my own use. As this is just a reference implementation, I will not be providing a crate, and instead will be copying the contents of [`src/lib.rs`](src/lib.rs) into my projects and modifying it as needed.

## Features

- Simple interface
- Grammar support
- Cross platform support (Windows, Linux, MacOS)
- GPU Acceleration via Vulkan & Metal
  - This means its cross-GPU compatible, working on AMD, Intel, and NVIDIA GPUs.

## Known Issues

- Crashes if the input or output exceeds the context window.
- Vulkan is [quite slow when compared to CUDA](https://github.com/ggml-org/llama.cpp/discussions/10879#discussioncomment-11600977), but I wanted a minimal, universal backend.
  - Certain models are particularly slow with Vulkan depending on quantization, see [this article](https://github.com/ggml-org/llama.cpp/wiki/Feature-matrix).
  - The less quantized a model is, the better it is for Vulkan. I got about a 10% improvement when using `Q8_0` instead of `Q4_K_M`.
- We should look into compiling in the [other backends](https://github.com/ggml-org/llama.cpp?tab=readme-ov-file#supported-backends) and switching between them at runtime based on hardware.
- The LLM will just run away with the prompt sometimes, endlessly generating more and more text, sometimes ending in a crash. This could be fixed but the CLI tool included with this repo is meant as an example of how to use the library so its not a priority. Tends to do this less when you give it more detailed instructions or enforce grammar.

## Building

```bash
git clone https://github.com/chand1012/llama-rs-example.git
cd llama-rs-example
cargo build --release
```

## Running

To run the program, use the following command:

```bash
cargo run -- [OPTIONS] <MODEL_PATH>
```

### Required Arguments

- `<MODEL_PATH>`: Path to the [GGUF model file](https://github.com/ggml-org/llama.cpp/tree/master?tab=readme-ov-file#obtaining-and-quantizing-models)

### Optional Arguments

- `--system-prompt <TEXT>`: Set the system prompt that defines the AI's behavior
  - Default: "You are a helpful AI assistant. Answer concisely and accurately."
- `--ctx-size <NUMBER>`: Set the maximum context length in tokens
- `-s, --seed <NUMBER>`: Set the RNG seed (default: 1234)
- `-t, --threads <NUMBER>`: Number of threads to use during generation (default: use all available threads)
- `--threads-batch <NUMBER>`: Number of threads to use during batch and prompt processing (default: use all available threads)
- `-v, --verbose`: Enable verbose llama.cpp logs
- `--grammar-file <FILE>`: Path to a grammar file in [GBNF format](https://github.com/ggml-org/llama.cpp/tree/master/grammars)
- `--disable-gpu`: Disable GPU acceleration
- `--stream`: Enable streaming output (shows tokens as they're generated)

### Examples

Basic usage with default settings:

```bash
cargo run -- path/to/model.gguf
```

Run with custom system prompt and streaming:

```bash
cargo run -- path/to/model.gguf --system-prompt "You are a helpful coding assistant." --stream
```

Run with specific thread count and context size:

```bash
cargo run -- path/to/model.gguf --threads 4 --ctx-size 2048
```

Use JSON grammar for structured output:

```bash
cargo run -- path/to/model.gguf --grammar-file grammars/json.gbnf
```

### Interactive Usage

Once running, you can:

1. Type your messages and press Enter to send them
2. The AI will respond based on the conversation history
3. Type 'exit' to quit the program

The chat history is maintained throughout the session, allowing for contextual conversations.
