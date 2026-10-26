$checkpoints = "D:/huggingface/modelos/UniAudio2/checkpoints"
$codecConfig = "tools/tokenizer/ReasoningCodec_film/infer_config.yaml"
$inputAudio = "samples/p225_002.wav"

python multi_task_inference.py `
  --task speech_edit `
  --stage all `
  --audio "$inputAudio" `
  --text "Make the speaker whisper throughout the recording, using a quiet, breathy voice. Preserve the spoken words." `
  --output_dir "./output/edit_speech/run1" `
  --llm_train_config "$checkpoints/llm_config.yaml" `
  --exp_dir "." `
  --resume "$checkpoints/llm_ep3.checkpoint" `
  --text_tokenizer_path "tools/tokenizer/Text2ID/llama3_2_tokenizer" `
  --prompt_text "Edit the speech according to the provided instruction." `
  --audio_tokenizer_config "$codecConfig" `
  --audio_model_path "$checkpoints/llm_ep3.checkpoint" `
  --codec_config "$codecConfig" `
  --codec_ckpt "$checkpoints/ReasoningCodec.checkpoint" `
  --codec_steps 10