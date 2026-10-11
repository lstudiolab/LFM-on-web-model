# Fine-tune Sora/LFM with LoRA

This path starts from a pretrained instruction model instead of random weights. The default base is [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), which supports multilingual chat and is released under Apache-2.0. Review the model card and dataset licenses before distributing a trained adapter.

## Recommended first run (GPU notebook)

1. Enable a GPU accelerator in your notebook runtime.
2. Clone this repository and install the dependencies:
   ```bash
   git clone https://github.com/lstudiolab/LFM-on-web-model.git
   cd LFM-on-web-model
   pip install -U "transformers>=4.46" "datasets>=3.0" "peft>=0.13" "accelerate>=1.0" "bitsandbytes>=0.44" safetensors
   ```
3. Put chat examples in JSONL at `data/sora_chat.jsonl`. Each line can be:
   ```json
   {"messages":[{"role":"user","content":"Hi!"},{"role":"assistant","content":"Hello! How can I help?"}]}
   ```
   The script also accepts `prompt` + `response`, or `instruction` + `output`.
4. Run a short smoke test first:
   ```bash
   python python/train_sora_lora.py --data data/sora_chat.jsonl --output /kaggle/working/sora-qwen-lora --max-steps 20 --limit 64 --save-steps 10
   ```
5. If the smoke test succeeds, train longer:
   ```bash
   python python/train_sora_lora.py --data data/sora_chat.jsonl --output /kaggle/working/sora-qwen-lora --max-steps 500 --save-steps 100
   ```

## Important

- The script requires a CUDA GPU and uses 4-bit QLoRA by default. `--no-4bit` disables quantization and will need substantially more GPU memory.
- The model's original weights are not modified. Only the LoRA adapter is saved.
- Download or copy `adapter-final/` from the notebook output to durable storage before the notebook session ends. Notebook working storage may be temporary.
- Start with clean, high-quality examples. More data is not automatically better; keep a separate evaluation set and inspect generated answers for repetition, instruction-following, and multilingual quality.
- Do not upload private conversations or personal data without permission.
- This is an initial training path, not proof that the resulting model is ready for production. Test generation and adapter loading before connecting it to the Sora/LFM web app.
