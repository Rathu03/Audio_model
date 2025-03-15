import jax.numpy as jnp
from whisper_jax import FlaxWhisperForConditionalGeneration, FlaxWhisperPipline

# path to the audio file to be transcribed
audio = "/path/to/audio.format"

transcribe = FlaxWhisperPipline("vasista22/whisper-tamil-medium", batch_size=16)
transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="ta", task="transcribe")

print('Transcription: ', transcribe(audio)["text"])
