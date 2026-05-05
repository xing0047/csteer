import torch as t
import matplotlib.pyplot as plt

_MODEL_PATHS = {
    "internvl3_5": {"8b": "../MODEL/InternVL3_5-8B"},
    "qwen3vl": {"8b": "../MODEL/Qwen3-VL-8B-Instruct"},
}


def set_plotting_settings():
    plt.style.use('seaborn-v0_8')
    params = {
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "font.family": "serif",
        "font.size": 13,
        "figure.autolayout": True,
        'figure.dpi': 600,
    }
    plt.rcParams.update(params)

    custom_colors = ['#377eb8', '#ff7f00', '#4daf4a',
                     '#f781bf', '#a65628', '#984ea3',
                     '#999999', '#e41a1c', '#dede00']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)


def add_vector_from_position(matrix, vector, position_ids, from_pos=None,
        is_qwen_vl=False, in_query_mask=None, in_decode_steer_flag=False):
    
    from_id = from_pos
    if from_id is None:
        from_id = position_ids.min().item() - 1

    if is_qwen_vl:
        if position_ids.shape[-1] > 1:
            # prefilling
            seq_1d_position_ids = t.arange(start=0, end=position_ids.shape[-1]).unsqueeze(0).to(position_ids.dtype).to(position_ids.device)
            mask = seq_1d_position_ids >= from_id
            if in_query_mask is not None:
                mask |= in_query_mask.bool()
        else:
            # decoding
            seq_1d_position_ids = t.arange(start=from_id, end=from_id+1).unsqueeze(0).to(position_ids.dtype).to(position_ids.device)
            mask = seq_1d_position_ids >= from_id
            if not in_decode_steer_flag:
                mask.fill_(False)
    else:
        mask = position_ids >= from_id
        if position_ids.shape[-1] > 1:
            # prefilling
            if in_query_mask is not None:
                mask |= in_query_mask.bool()
        else:
            # decoding
            if not in_decode_steer_flag:
                mask.fill_(False)
    # print(mask)
    # import pdb; pdb.set_trace()
    mask = mask.unsqueeze(-1)
    matrix += mask.float() * vector
    return matrix


def find_last_subtensor_position(tensor, sub_tensor):
    n, m = tensor.size(0), sub_tensor.size(0)
    if m > n:
        return -1
    for i in range(n - m, -1, -1):
        if t.equal(tensor[i : i + m], sub_tensor):
            return i
    return -1


def find_instruction_end_postion(tokens, end_str):
    start_pos = find_last_subtensor_position(tokens, end_str)
    if start_pos == -1:
        return -1
    return start_pos + len(end_str) - 1


def make_tensor_save_suffix(layer, model_name, model_size):
    return f'{str(layer).zfill(2)}_{model_name}_{model_size}'


def get_model_path(model_name: str, model_size: str):
    return _MODEL_PATHS[model_name][model_size]


_model_to_wrapper_map = None


def get_model_to_wrapper_map():
    """Lazily built so wrappers.* can import utils.helpers without a cycle."""
    global _model_to_wrapper_map
    if _model_to_wrapper_map is None:
        from wrappers.internvl3_5_wrapper import InternVL3_5_Wrapper
        from wrappers.qwen3vl_wrapper import Qwen3VL_Wrapper
        from wrappers.openai_wrapper import OpenAIWrapper

        _model_to_wrapper_map = {
            "internvl3_5": {"8b": InternVL3_5_Wrapper},
            "qwen3vl": {"8b": Qwen3VL_Wrapper},
            "gpt-4o": OpenAIWrapper,
            "gemini-2.5-pro": OpenAIWrapper,
            "o3": OpenAIWrapper,
        }
    return _model_to_wrapper_map