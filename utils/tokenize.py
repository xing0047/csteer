from .conversation import get_conv_template

ADD_FROM_POS_CHAT = "<|im_start|>assistant\n"

def tokenizer_internvl_chat(
    model, tokenizer, question, model_output,
    pixel_values=None, history=None, num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False
):
    if history is None and pixel_values is not None and '<image>' not in question:
        question = '<image>\n' + question

    if num_patches_list is None:
        num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    template = get_conv_template(model.template)
    template.system_message = model.system_message

    history = [] if history is None else history
    for (old_question, old_answer) in history:
        template.append_message(template.roles[0], old_question)
        template.append_message(template.roles[1], old_answer)
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    if verbose and pixel_values is not None:
        image_bs = pixel_values.shape[0]
        print(f'dynamic ViT batch size: {image_bs}')

    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace('<image>', image_tokens, 1)
    
    if model_output is not None:
        query += f" {model_output.strip()}"
    
    return tokenizer.encode(query)

def find_marker_pos(tokens, tokenizer):
    markers_pos = []
    assert len(tokens.shape) == 2
    decoded_tokens = [tokenizer.decode(token) for token in tokens[0]]
    marker_flag = False
    for token_pos, token in enumerate(decoded_tokens):
        if '[' in token:
            marker_flag = True
        if ']' in token:
            marker_str += token
            marker_pos.append(token_pos)
            markers_pos.append((marker_str, marker_pos))
            marker_flag = False
        if not marker_flag:
            marker_str, marker_pos = "", []
        if marker_flag:
            marker_str += token
            marker_pos.append(token_pos)
    if len(markers_pos) > 3:
        return markers_pos[3:]
    else:
        return None
