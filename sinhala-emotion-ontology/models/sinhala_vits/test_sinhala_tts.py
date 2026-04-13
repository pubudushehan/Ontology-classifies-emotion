from TTS.api import TTS

# Load local model
tts = TTS(model_path="models/sinhala_vits/Roshan_270000.pth",
          config_path="models/sinhala_vits/Roshan_config.json")

# Generate
tts.tts_to_file("ආයුබෝවන්! මගේ emotion TTS prototype!", "demo_sinhala.wav")
print("✅ Sinhala audio generated!")