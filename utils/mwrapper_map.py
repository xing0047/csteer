from .wrappers.internvl3_5_wrapper import InternVL3_5_Wrapper
from .wrappers.qwen3vl_wrapper import Qwen3VL_Wrapper
from .wrappers.openai_wrapper import OpenAIWrapper

model_to_wrapper_map = {
    'internvl3_5': {
        '8b': InternVL3_5_Wrapper,
    },
    'qwen3vl': {
        '8b': Qwen3VL_Wrapper,
    },
    'gpt-4o': OpenAIWrapper,
    'gemini-2.5-pro': OpenAIWrapper,
    'o3': OpenAIWrapper,
}
