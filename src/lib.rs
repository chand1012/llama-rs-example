use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};
use std::num::NonZeroU32;
use std::path::PathBuf;

pub struct Llama {
    model: Box<LlamaModel>,
    ctx: llama_cpp_2::context::LlamaContext<'static>,
    sampler: LlamaSampler,
    batch: LlamaBatch,
    use_grammar: bool, // probably a better way to check but good enough for now
    embedding: bool,
}

impl Llama {
    pub const JSON_GRAMMAR: &'static str = include_str!("../grammars/json.gbnf");
    /// Create a new Llama instance
    pub fn new(
        model_path: PathBuf,
        ctx_size: Option<NonZeroU32>,
        n_threads: Option<i32>,
        n_batch: Option<i32>,
        seed: Option<u32>,
        grammar: Option<&str>,
        embedding: bool,
        verbose: bool,
        _disable_gpu: bool,
    ) -> Result<Self> {
        // Initialize logging if verbose
        if verbose {
            send_logs_to_tracing(LogOptions::default().with_logs_enabled(true));
        }

        // Initialize backend
        let backend = LlamaBackend::init()?;

        // Create model parameters
        // if we're on windows or linux and GPU is not disabled, we need to force gpu
        #[cfg(any(target_os = "windows", target_os = "linux"))]
        let model_params = if !_disable_gpu {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        } else {
            LlamaModelParams::default()
        };
        #[cfg(not(any(target_os = "windows", target_os = "linux")))]
        let model_params = LlamaModelParams::default();

        // Load the model
        let model = Box::new(
            LlamaModel::load_from_file(&backend, model_path, &model_params)
                .context("Failed to load model")?,
        );

        // Initialize context parameters
        let mut ctx_params = LlamaContextParams::default()
            .with_n_threads(
                n_threads.unwrap_or(std::thread::available_parallelism()?.get().try_into()?),
            )
            .with_n_threads_batch(n_batch.unwrap_or(
                n_threads.unwrap_or(std::thread::available_parallelism()?.get().try_into()?),
            ))
            .with_n_ctx(ctx_size.or(Some(NonZeroU32::new(2048).unwrap())));

        if embedding {
            ctx_params = ctx_params.with_embeddings(true);
        }

        // Create context
        let ctx = unsafe {
            std::mem::transmute(
                model
                    .new_context(&backend, ctx_params)
                    .context("Failed to create context")?,
            )
        };

        let sampler = match grammar {
            Some(grammar) => LlamaSampler::chain_simple([
                LlamaSampler::grammar(&model, grammar, "root"),
                LlamaSampler::dist(seed.unwrap_or(1234)),
                LlamaSampler::greedy(),
            ]),
            None => LlamaSampler::chain_simple([
                LlamaSampler::dist(seed.unwrap_or(1234)),
                LlamaSampler::greedy(),
            ]),
        };

        // Create batch for token processing
        let batch = LlamaBatch::new(512, 1);

        Ok(Self {
            model,
            ctx,
            sampler,
            batch,
            use_grammar: grammar.is_some(),
            embedding,
        })
    }

    /// Generate text from a prompt
    pub fn generate(&mut self, prompt: &str, max_tokens: i32) -> Result<String> {
        if self.embedding {
            anyhow::bail!("Model is not configured for text generation");
        }
        // Tokenize the prompt
        let tokens_list = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .context("Failed to tokenize prompt")?;

        // Verify context size
        let n_ctx = self.ctx.n_ctx() as i32;
        let n_kv_req = tokens_list.len() as i32 + (max_tokens - tokens_list.len() as i32);

        if n_kv_req > n_ctx {
            anyhow::bail!("Required KV cache size is too large for context window");
        }

        if tokens_list.len() >= usize::try_from(max_tokens)? {
            anyhow::bail!("Prompt is longer than maximum allowed tokens");
        }

        // Clear and prepare batch
        self.batch.clear();

        // Process prompt tokens
        let last_index = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            let is_last = i == last_index;
            self.batch.add(token, i, &[0], is_last)?;
        }

        // Initial decode
        self.ctx
            .decode(&mut self.batch)
            .context("Failed to decode. Input may exceed max tokens.")?;

        let mut n_cur = self.batch.n_tokens();
        let mut output = String::new();
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        // Generation loop
        while n_cur <= max_tokens {
            // Sample next token
            let token = self.sampler.sample(&self.ctx, self.batch.n_tokens() - 1);

            // only run this if the grammar is None
            // see this issue for more info: https://github.com/utilityai/llama-cpp-rs/issues/604#issuecomment-2562298232
            if !self.use_grammar {
                // the grammar sampler handles this for us when it's enabled
                // so we only need to handle it when not using grammar
                self.sampler.accept(token);
            }

            // Check for end of generation
            if self.model.is_eog_token(token) {
                break;
            }

            // Convert token to text
            let output_bytes = self.model.token_to_bytes(token, Special::Tokenize)?;
            let mut token_text = String::with_capacity(32);
            let _ = decoder.decode_to_string(&output_bytes, &mut token_text, false);
            output.push_str(&token_text);

            // Prepare next iteration
            self.batch.clear();
            self.batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;

            // Decode
            self.ctx
                .decode(&mut self.batch)
                .context("Failed to decode. Output may exceed max tokens.")?;
        }

        Ok(output)
    }

    /// Get the model's tokenizer
    pub fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        Ok(self
            .model
            .str_to_token(text, AddBos::Always)
            .context("Failed to tokenize text")?
            .into_iter()
            .map(|t| t.0)
            .collect())
    }

    /// Convert tokens back to text
    pub fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        let mut text = String::new();
        for &token in tokens {
            let llama_token = llama_cpp_2::token::LlamaToken(token);
            text.push_str(&self.model.token_to_str(llama_token, Special::Tokenize)?);
        }
        Ok(text)
    }

    /// Generate an embedding
    /// Returns a vector of floats
    /// The length of the vector is the embedding dimension
    /// The embedding dimension is specified in the model parameters
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        if !self.embedding {
            anyhow::bail!("Model is not configured for embeddings");
        }
        let tokens = self
            .model
            .str_to_token(text, AddBos::Always)
            .with_context(|| format!("Failed to tokenize {text}"))?;
        let n_ctx = self.ctx.n_ctx() as usize;

        if tokens.len() > n_ctx {
            anyhow::bail!("Text is longer than maximum allowed tokens");
        }

        self.batch.clear();

        // Add tokens to batch with sequence ID 0
        self.batch.add_sequence(&tokens, 0, false)?;

        // Clear KV cache and decode
        self.ctx.clear_kv_cache();
        self.ctx
            .decode(&mut self.batch)
            .context("Failed to decode batch")?;

        // Get embeddings for the sequence
        let embeddings = self
            .ctx
            .embeddings_seq_ith(0)
            .context("Failed to get embeddings")?;

        Ok(embeddings.to_vec())
    }

    /// Generate embeddings for multiple texts in batches
    /// Returns a vector of embeddings, preserving the order of input texts
    /// Each embedding is a vector of floats with length equal to the model's embedding dimension
    pub fn batch_embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if !self.embedding {
            anyhow::bail!("Model is not configured for embeddings");
        }

        let n_ctx = self.ctx.n_ctx() as usize;
        let mut output = Vec::with_capacity(texts.len());

        // Tokenize all texts first
        let tokens_list = texts
            .iter()
            .map(|text| self.model.str_to_token(text, AddBos::Always))
            .collect::<Result<Vec<_>, _>>()
            .with_context(|| "Failed to tokenize texts")?;

        // Check if any text exceeds context size
        if tokens_list.iter().any(|tokens| tokens.len() > n_ctx) {
            anyhow::bail!("One or more texts exceed the maximum allowed tokens");
        }

        let mut max_seq_id = 0;

        // Process texts in batches that fit within context window
        for tokens in &tokens_list {
            // If adding these tokens would exceed context size, process current batch first
            if (self.batch.n_tokens() as usize + tokens.len()) > n_ctx {
                // Clear KV cache and decode current batch
                self.ctx.clear_kv_cache();
                self.ctx
                    .decode(&mut self.batch)
                    .context("Failed to decode batch")?;

                // Get embeddings for all sequences in current batch
                for seq_id in 0..max_seq_id {
                    let embedding = self.ctx.embeddings_seq_ith(seq_id).with_context(|| {
                        format!("Failed to get embeddings for sequence {seq_id}")
                    })?;
                    output.push(embedding.to_vec());
                }

                // Clear batch for next round
                self.batch.clear();
                max_seq_id = 0;
            }

            // Add new sequence to batch
            self.batch.add_sequence(tokens, max_seq_id, false)?;
            max_seq_id += 1;
        }

        // Process final batch if there are any remaining sequences
        if max_seq_id > 0 {
            self.ctx.clear_kv_cache();
            self.ctx
                .decode(&mut self.batch)
                .context("Failed to decode final batch")?;

            for seq_id in 0..max_seq_id {
                let embedding = self
                    .ctx
                    .embeddings_seq_ith(seq_id)
                    .with_context(|| format!("Failed to get embeddings for sequence {seq_id}"))?;
                output.push(embedding.to_vec());
            }
        }

        Ok(output)
    }

    /// Stream text from a prompt, calling the provided callback function for each token generated
    /// Returns the complete generated text as a string
    pub fn stream<F>(&mut self, prompt: &str, max_tokens: i32, mut callback: F) -> Result<String>
    where
        F: FnMut(&str) -> Result<()>,
    {
        if self.embedding {
            anyhow::bail!("Model is not configured for text generation");
        }
        // Tokenize the prompt
        let tokens_list = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .context("Failed to tokenize prompt")?;

        // Verify context size
        let n_ctx = self.ctx.n_ctx() as i32;
        let n_kv_req = tokens_list.len() as i32 + (max_tokens - tokens_list.len() as i32);

        if n_kv_req > n_ctx {
            anyhow::bail!("Required KV cache size is too large for context window");
        }

        if tokens_list.len() >= usize::try_from(max_tokens)? {
            anyhow::bail!("Prompt is longer than maximum allowed tokens");
        }

        // Clear and prepare batch
        self.batch.clear();

        // Process prompt tokens
        let last_index = (tokens_list.len() - 1) as i32;
        for (i, token) in (0_i32..).zip(tokens_list.into_iter()) {
            let is_last = i == last_index;
            self.batch.add(token, i, &[0], is_last)?;
        }

        // Initial decode
        self.ctx
            .decode(&mut self.batch)
            .context("Failed to decode. Input may exceed max tokens.")?;

        let mut n_cur = self.batch.n_tokens();
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut output = String::new();

        // Generation loop
        while n_cur <= max_tokens {
            // Sample next token
            let token = self.sampler.sample(&self.ctx, self.batch.n_tokens() - 1);

            // only run this if the grammar is None
            if !self.use_grammar {
                self.sampler.accept(token);
            }

            // Check for end of generation
            if self.model.is_eog_token(token) {
                break;
            }

            // Convert token to text and stream it
            let output_bytes = self.model.token_to_bytes(token, Special::Tokenize)?;
            let mut token_text = String::with_capacity(32);
            let _ = decoder.decode_to_string(&output_bytes, &mut token_text, false);
            callback(&token_text)?;
            output.push_str(&token_text);

            // Prepare next iteration
            self.batch.clear();
            self.batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;

            // Decode
            self.ctx
                .decode(&mut self.batch)
                .context("Failed to decode. Output may exceed max tokens.")?;
        }

        Ok(output)
    }
}
