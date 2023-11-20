from scipy import io
from transformers import AutoProcessor, BarkModel

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small")

voice_preset = "v2/en_speaker_6"

inputs = processor("Models won't be available and only tokenizers, configuration and file/data utilities can be used.", voice_preset=voice_preset)

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()


sampling_rate = model.generation_config.sample_rate
io.wavfile.write("bark_out.wav", rate=sampling_rate, data=audio_array)