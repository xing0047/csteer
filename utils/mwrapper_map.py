from .wrappers.internvl3_5_wrapper import InternVL3_5_Wrapper
from .wrappers.qwen3vl_wrapper import Qwen3VL_Wrapper
from .wrappers.internvl3_wrapper import InternVL3_Wrapper
from .wrappers.openai_wrapper import OpenAIWrapper

model_to_wrapper_map = {
    'internvl3_5': {
        '2b': InternVL3_5_Wrapper,
        '8b': InternVL3_5_Wrapper,
        '38b': InternVL3_5_Wrapper,
    },
    'qwen3vl': {
        '2b': Qwen3VL_Wrapper,
        '8b': Qwen3VL_Wrapper,
        '32b': Qwen3VL_Wrapper,
    },
    'internvl3': {
        '2b': InternVL3_Wrapper,
        '8b': InternVL3_Wrapper,
        '38b': InternVL3_Wrapper,
    },
    'gpt-4o': OpenAIWrapper,
    'gemini-2.5-pro': OpenAIWrapper,
    'o3': OpenAIWrapper,
}