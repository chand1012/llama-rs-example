//! This is a translation of simple.cpp in llama.cpp using llama-cpp-2.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use hf_hub::api::sync::ApiBuilder;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{ggml_time_us, send_logs_to_tracing, LogOptions};
use rustyline::DefaultEditor;

use std::ffi::CString;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::pin;
use std::str::FromStr;
use std::time::Duration;

#[derive(clap::Subcommand, Debug, Clone)]
enum Model {
    /// Use an already downloaded model
    Local {
        /// The path to the model
        path: PathBuf,
    },
    /// Download a model from huggingface
    #[clap(name = "hf-model")]
    HuggingFace {
        /// the repo containing the model
        repo: String,
        /// the model name
        model: String,
    },
}

impl Model {
    /// Convert the model to a path - may download from huggingface
    fn get_or_load(self) -> Result<PathBuf> {
        match self {
            Model::Local { path } => Ok(path),
            Model::HuggingFace { model, repo } => ApiBuilder::new()
                .with_progress(true)
                .build()
                .with_context(|| "unable to create huggingface api")?
                .model(repo)
                .get(&model)
                .with_context(|| "unable to download model"),
        }
    }
}

/// Parse a single key-value pair
fn parse_key_val(s: &str) -> Result<(String, ParamOverrideValue)> {
    let pos = s
        .find('=')
        .ok_or_else(|| anyhow!("invalid KEY=value: no `=` found in `{}`", s))?;
    let key = s[..pos].parse()?;
    let value: String = s[pos + 1..].parse()?;
    let value = i64::from_str(&value)
        .map(ParamOverrideValue::Int)
        .or_else(|_| f64::from_str(&value).map(ParamOverrideValue::Float))
        .or_else(|_| bool::from_str(&value).map(ParamOverrideValue::Bool))
        .map_err(|_| anyhow!("must be one of i64, f64, or bool"))?;

    Ok((key, value))
}

#[derive(clap::Parser, Debug, Clone)]
struct Args {
    /// The path to the model
    #[command(subcommand)]
    model: Model,
    /// The system prompt that defines the AI's behavior
    #[clap(
        long,
        default_value = "You are a helpful AI assistant. Answer concisely and accurately."
    )]
    system_prompt: String,
    /// set the maximum context length in tokens
    #[arg(long, default_value_t = 2048)]
    ctx_size: u32,
    /// override some parameters of the model
    #[arg(short = 'o', value_parser = parse_key_val)]
    key_value_overrides: Vec<(String, ParamOverrideValue)>,
    /// Disable offloading layers to the gpu
    #[cfg(any(feature = "cuda", feature = "vulkan"))]
    #[clap(long)]
    disable_gpu: bool,
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
}

fn generate_response(
    ctx: &mut llama_cpp_2::context::LlamaContext,
    model: &LlamaModel,
    prompt: &str,
    sampler: &mut LlamaSampler,
) -> Result<String> {
    let tokens_list = model
        .str_to_token(prompt, AddBos::Always)
        .context("failed to tokenize prompt")?;

    let mut batch = LlamaBatch::new(512, 1);
    let last_index: i32 = (tokens_list.len() - 1) as i32;

    for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
        let is_last = i == last_index;
        batch.add(token, i, &[0], is_last)?;
    }

    ctx.decode(&mut batch).context("initial decode failed")?;

    let mut response = String::new();
    let mut n_cur = batch.n_tokens();
    let mut decoder = encoding_rs::UTF_8.new_decoder();

    loop {
        let token = sampler.sample(ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let output_bytes = model.token_to_bytes(token, Special::Tokenize)?;
        let mut output_string = String::with_capacity(32);
        let _decode_result = decoder.decode_to_string(&output_bytes, &mut output_string, false);

        print!("{}", output_string);
        std::io::stdout().flush()?;
        response.push_str(&output_string);

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;

        ctx.decode(&mut batch).context("failed to eval")?;
    }

    println!();
    Ok(response)
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.verbose {
        tracing_subscriber::fmt().init();
    }
    send_logs_to_tracing(LogOptions::default().with_logs_enabled(args.verbose));

    // Initialize backend
    let backend = LlamaBackend::init()?;

    // Configure model parameters
    let mut model_params = { LlamaModelParams::default() };

    let mut model_params = pin!(model_params);

    for (k, v) in &args.key_value_overrides {
        let k = CString::new(k.as_bytes()).context("invalid key")?;
        model_params.as_mut().append_kv_override(k.as_c_str(), *v);
    }

    // Load the model
    println!("Loading model...");
    let model_path = args.model.get_or_load().context("failed to get model")?;
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .context("unable to load model")?;

    // Initialize context
    let ctx_size = NonZeroU32::new(args.ctx_size).unwrap();
    let mut ctx_params = LlamaContextParams::default().with_n_ctx(Some(ctx_size));

    if let Some(threads) = args.threads {
        ctx_params = ctx_params.with_n_threads(threads);
    }
    if let Some(threads_batch) = args.threads_batch.or(args.threads) {
        ctx_params = ctx_params.with_n_threads_batch(threads_batch);
    }

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .context("unable to create the llama_context")?;

    // Initialize the chat interface
    let mut rl = DefaultEditor::new()?;
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::dist(args.seed.unwrap_or(1234)),
        LlamaSampler::greedy(),
    ]);

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

                let response = generate_response(&mut ctx, &model, &chat_history, &mut sampler)?;
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
