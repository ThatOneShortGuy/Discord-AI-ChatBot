from typing import Generator, Literal, Optional, Union

from exllamav2 import (ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config,
                       ExLlamaV2Tokenizer)
from exllamav2.embedding import ExLlamaV2Embedding
from exllamav2.generator import ExLlamaV2Sampler, ExLlamaV2StreamingGenerator
from huggingface_hub import hf_hub_download, snapshot_download
from llama_cpp import Llama
from torch import Tensor


class ModelHandler:
    def __init__(self, hf_model_name: str, model_type: Union[Literal['exl2'], Literal['gguf']], file_path: Optional[str] = None, revision: Optional[str] = None, max_context_size: Optional[int] = None):
        self.model_name = hf_model_name
        self.model_type = model_type
        if file_path is not None:
            self.model_path = hf_hub_download(hf_model_name, file_path)
        elif revision is not None:
            self.model_path = snapshot_download(
                hf_model_name, revision=revision)
        else:
            raise ValueError('Either file_path or revision must be provided')
        self.model_type = model_type
        self.max_context_size = max_context_size
        self.model = None
        self.tokenizer = None

    def __call__(self, text: str, max_tokens: Optional[int], stop: Optional[list[str]] = None, stream: Optional[bool] = False, **kwargs) -> Union[Generator[str, None, None], str]:
        if self.model_type == 'exl2':
            return self._exl2_generate(text, max_tokens, stop, stream, **kwargs)
        elif self.model_type == 'gguf':
            return self._gguf_generate(text, max_tokens, stop, stream, **kwargs)
        else:
            raise ValueError('Invalid model type')

    def _gguf_generate(self, text: str, max_tokens: Optional[int], stop: Optional[list[str]], stream: Optional[bool], **kwargs) -> Union[Generator[str, None, None], str]:
        if self.model is None:
            raise ValueError('Model not loaded. Call load_model() first')
        if not isinstance(self.model, Llama):
            raise ValueError('Invalid model type')
        if stream:
            gen = self.model(text, max_tokens=max_tokens,
                             stop=stop, stream=True, **kwargs)
            for response in gen:
                yield response['choices'][0]['text']  # type: ignore
        else:
            return self.model(text, max_tokens=max_tokens, stop=stop, **kwargs)['choices'][0]['text'] # type: ignore

    def _exl2_generate(self, text: str, max_tokens: Optional[int], stop: Optional[list[str]], stream: Optional[bool], **kwargs) -> Union[Generator[str, None, None], str]:
        if self.model is None:
            raise ValueError('Model not loaded. Call load_model() first')
        if not isinstance(self.model, ExLlamaV2):
            raise ValueError('Invalid model type')
        if stream:
            input_ids: Tensor = self.tokenizer.encode(text)  # type: ignore
            self.generator.begin_stream(input_ids, self.settings)
            num_tokens = input_ids.shape[-1]
            while True:
                chunk, eos, _ = self.generator.stream()
                yield chunk
                if eos or (max_tokens is not None and num_tokens >= max_tokens):
                    break
                num_tokens += 1
        else:
            return self.generator.generate_simple(text, self.settings, max_tokens if max_tokens is not None else self.model.config.max_seq_len, **kwargs)

    def load_model(self):
        print(f'Loading model: {self.model_path}')
        if self.model_type == 'exl2':
            self._load_exl2()
        elif self.model_type == 'gguf':
            self._load_gguf()
        else:
            raise ValueError('Invalid model type')

    def tokenize(self, text: Union[str, bytes]) -> Tensor:
        if self.tokenizer is None:
            raise ValueError('Model not loaded. Call load_model() first')
        return self.tokenizer.encode(str(text))[-1]  # type: ignore

    def _load_exl2(self):
        config = ExLlamaV2Config()
        config.model_dir = self.model_path
        print('Preparing model...')
        config.prepare()
        if self.max_context_size is not None:
            config.max_seq_len = self.max_context_size
            config.max_input_len = 784
        print('Model prepared!')

        self.model = ExLlamaV2(config)
        print("Loading model: " + self.model_path)

        cache = ExLlamaV2Cache(self.model, lazy=True)
        self.model.load_autosplit(cache)
        print("Loaded!")
        print(f'Max context length: {self.model.config.max_seq_len}')

        # Initialize tokenizer
        self.tokenizer = ExLlamaV2Tokenizer(config)

        # Initialize generator
        self.generator = ExLlamaV2StreamingGenerator(
            self.model, cache, self.tokenizer)
        self.generator.warmup()

        # Settings
        self.settings = ExLlamaV2Sampler.Settings()
        self.max_context_size = self.model.config.max_seq_len

    def _load_gguf(self):
        raise NotImplementedError('GGUF model not implemented yet')

    def embed(self, text: str):
        if self.model is None:
            raise ValueError('Model not loaded. Call load_model() first')
        if self.model_type == 'exl2':
            inputs_ids: Tensor = self.tokenizer.encode(text)  # type: ignore
            embedding_layer: ExLlamaV2Embedding = self.model.modules_dict['model.embed_tokens']
            return embedding_layer.forward(inputs_ids)
        elif self.model_type == 'gguf':
            raise NotImplementedError('GGUF model not implemented yet')
        else:
            raise ValueError('Invalid model type')

    def get_bos(self):
        if self.model is None:
            raise ValueError('Model not loaded. Call load_model() first')
        if self.model_type == 'exl2':
            return self.tokenizer.bos_token  # type: ignore
        elif self.model_type == 'gguf':
            raise NotImplementedError('GGUF model not implemented yet')
        else:
            raise ValueError('Invalid model type')

    def get_eos(self):
        if self.model is None:
            raise ValueError('Model not loaded. Call load_model() first')
        if self.model_type == 'exl2':
            return self.tokenizer.eos_token  # type: ignore
        elif self.model_type == 'gguf':
            raise NotImplementedError('GGUF model not implemented yet')
