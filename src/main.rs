//! This is a translation of simple.cpp in llama.cpp using llama-cpp-2.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use anyhow::{Context, Result};
use clap::Parser;
use rustyline::DefaultEditor;
use std::fs;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;

#[derive(clap::Parser, Debug, Clone)]
struct Args {
    /// The path to the model file
    #[arg(help = "Path to the GGUF model file")]
    model_path: PathBuf,
    /// The system prompt that defines the AI's behavior
    #[clap(
        long,
        default_value = "You are a helpful AI assistant. Answer concisely and accurately."
    )]
    system_prompt: String,
    /// set the maximum context length in tokens
    #[arg(long)]
    ctx_size: Option<u32>,
    #[arg(short = 's', long, help = "RNG seed (default: 1234)")]
    seed: Option<u32>,
    #[arg(
        short = 't',
        long,
        help = "number of threads to use during generation (default: use all available threads)"
    )]
    threads: Option<i32>,
    #[arg(
        long,
        help = "number of threads to use during batch and prompt processing (default: use all available threads)"
    )]
    threads_batch: Option<i32>,
    #[arg(short = 'v', long, help = "enable verbose llama.cpp logs")]
    verbose: bool,
    #[arg(long, help = "path to a grammar file in GBNF format")]
    grammar_file: Option<PathBuf>,
    #[arg(long, help = "disable GPU acceleration")]
    disable_gpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load the model
    println!("Loading model...");

    // Load grammar if specified
    let grammar = if let Some(grammar_path) = args.grammar_file {
        Some(fs::read_to_string(grammar_path).context("failed to read grammar file")?)
    } else {
        None
    };

    // Convert ctx_size to NonZeroU32
    let ctx_size = args.ctx_size.map(NonZeroU32::new).map(Option::unwrap);

    // Initialize Llama
    let mut llama = llama_rs_example::Llama::new(
        args.model_path,
        ctx_size,
        args.threads,
        args.threads_batch,
        args.seed,
        grammar.as_deref(),
        args.verbose,
        args.disable_gpu,
    )?;

    // Initialize the chat interface
    let mut rl = DefaultEditor::new()?;
    println!("Model loaded successfully! Type your messages and press Enter. Type 'exit' to quit.");
    println!("System prompt: {}", args.system_prompt);

    let mut chat_history = args.system_prompt;
    chat_history.push('\n');

    loop {
        let readline = rl.readline("You: ");
        match readline {
            Ok(line) => {
                if line.trim().eq_ignore_ascii_case("exit") {
                    println!("Goodbye!");
                    break;
                }

                rl.add_history_entry(line.as_str())?;

                chat_history.push_str("User: ");
                chat_history.push_str(&line);
                chat_history.push_str("\nAssistant: ");

                print!("Assistant: ");
                std::io::stdout().flush()?;

                // Use a reasonable max_tokens value that fits within context
                let response = llama.generate(&chat_history, 2048)?;
                print!("{}", response);
                println!();

                chat_history.push_str(&response);
                chat_history.push('\n');
            }
            Err(err) => {
                println!("Error: {}", err);
                break;
            }
        }
    }

    Ok(())
}
