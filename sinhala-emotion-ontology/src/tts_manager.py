import os
import io

class TTSManager:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.base_dir, "models")
        
        # Cache for loaded models
        self.loaded_models = {}
        
        # Registry of available models
        self.available_models = {
            "v3": {
                "type": "coqui",
                "model_path": os.path.join(self.models_dir, "sinhala_vits", "sinhala_emo_tts_v3.pth"),
                "config_path": os.path.join(self.models_dir, "sinhala_vits", "config.json"),
            }
        }
        
    def _get_model(self, model_id: str):
        if model_id not in self.available_models:
            raise ValueError(f"Model ID '{model_id}' is not registered.")
            
        if model_id not in self.loaded_models:
            model_info = self.available_models[model_id]
            print(f"Loading TTS model '{model_id}'...")
            
            from TTS.utils.synthesizer import Synthesizer
            import torch
            use_cuda = torch.cuda.is_available()
            synth = Synthesizer(
                tts_checkpoint=model_info["model_path"],
                tts_config_path=model_info["config_path"],
                use_cuda=use_cuda
            )
            self.loaded_models[model_id] = synth
                
            print(f"Loaded '{model_id}' successfully.")
            
        return self.loaded_models[model_id]
        
    def generate_audio(self, text: str, model_id: str = "v3") -> bytes:
        """
        Generates audio using the specified model and returns it as WAV bytes.
        """
        model = self._get_model(model_id)
        
        from src.sinhala_to_roman import sinhala_to_roman
        roman_text = sinhala_to_roman(text)
        print(f"Romanized text: {roman_text}")
        
        wav = model.tts(roman_text)
        out = io.BytesIO()
        model.save_wav(wav, out)
        out.seek(0)
        return out.read()
