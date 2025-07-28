import numpy as np
import re
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .asr_interface import ASRInterface


class VoiceRecognition(ASRInterface):
    """MERaLiON ASR implementation using Hugging Face models."""

    def __init__(self, model_path: str, device: str = "auto") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            use_safetensors=True,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

    def transcribe_np(self, audio: np.ndarray) -> str:
        audio_list = [audio]
        prompt_template = "Instruction: {query} \nFollow the text instruction based on the following audio: <SpeechHere>"
        transcribe_prompt = "Please transcribe this speech."
        conversation = [
            [
                {
                    "role": "user",
                    "content": prompt_template.format(query=transcribe_prompt),
                }
            ]
        ]
        chat_prompt = self.processor.tokenizer.apply_chat_template(
            conversation=conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=chat_prompt, audios=audio_list)
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.device)
                if value.dtype == torch.float32 and self.device == "cuda":
                    inputs[key] = inputs[key].to(torch.float16)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            num_beams=1,
        )
        generated_ids = outputs[:, inputs["input_ids"].size(1) :]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        text = response[0].strip()
        # Remove common prefixes like "Assistant:" that the model may generate
        text = re.sub(r"^\s*\w+:\s*", "", text)
        if text.startswith(":"):
            text = text[1:].lstrip()
        return text
