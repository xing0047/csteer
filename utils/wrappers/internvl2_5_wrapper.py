import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from utils.helpers import add_vector_from_position, find_instruction_end_postion, get_model_path
from utils.tokenize import (
    tokenizer_internvl_chat,
    ADD_FROM_POS_CHAT,
)
from typing import Optional
from utils.conversation import get_conv_template
import pdb

class AttnWrapper(t.nn.Module):
    """
    Wrapper for attention mechanism to save activations
    """

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output


class BlockOutputWrapper(t.nn.Module):
    """
    Wrapper for block to save activations and unembed them
    """

    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer
        self.block.attention = AttnWrapper(self.block.attention)
        self.feed_forward = self.block.feed_forward
        self.attention_norm = self.block.attention_norm
        self.ffn_norm = self.block.ffn_norm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None
        self.from_position = None

        self.save_internal_decodings = False

        self.calc_dot_product_with = None
        self.dot_products = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = t.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = t.dot(last_token_activations, self.calc_dot_product_with) / (
                t.norm(last_token_activations) * t.norm(self.calc_dot_product_with)
            )
            self.dot_products.append((top_token, dot_product.cpu().item()))
        if self.add_activations is not None:
            augmented_output = add_vector_from_position(
                matrix=output[0],
                vector=self.add_activations,
                position_ids=kwargs["position_ids"],
                from_pos=self.from_position,
            )
            output = (augmented_output,) + output[1:]

        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        self.add_activations = None
        self.activations = None
        self.block.attention.activations = None
        self.from_position = None
        self.calc_dot_product_with = None
        self.dot_products = []


class InternVL2_5_Wrapper:
    def __init__(
        self,
        size: str = "8b",
        override_model_weights_path: Optional[str] = None,
    ):
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.model_name_path = get_model_path(size)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_path, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_path, torch_dtype=t.bfloat16, low_cpu_mem_usage=True,
            use_flash_attn=True, trust_remote_code=True
        )
        if override_model_weights_path is not None:
            self.model.load_state_dict(t.load(override_model_weights_path))
        self.model = self.model.to(self.device)
        self.END_STR = t.tensor(self.tokenizer.encode(ADD_FROM_POS_CHAT)[1:]).to(
            self.device
        )
        self.template = get_conv_template(self.model.template)
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.template.sep.strip())
        for i, layer in enumerate(self.model.language_model.model.layers):
            self.model.language_model.model.layers[i] = BlockOutputWrapper(
                layer, self.model.language_model.output, self.model.language_model.model.norm, self.tokenizer
            )

    def set_save_internal_decodings(self, value: bool):
        for layer in self.model.language_model.model.layers:
            layer.save_internal_decodings = value

    def set_from_positions(self, pos: int):
        for layer in self.model.language_model.model.layers:
            layer.from_position = pos

    def generate(self, tokens, inputs_embeds, **kwargs):
        with t.no_grad():
            instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            self.set_from_positions(instr_pos)
            kwargs.update(
                {'eos_token_id': self.eos_token_id}
            )
            generated = self.model.language_model.generate(
                inputs_embeds=inputs_embeds, **kwargs
            )
            generated_str = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            generated_str = generated_str.split(self.template.sep.strip())[0].strip()
            return generated_str

    def generate_text(
        self, 
        pixel_values, 
        question,
        model_output: Optional[str] = None, 
        **kwargs
    ) -> str:
        input_ids = tokenizer_internvl_chat(
            self.model, self.tokenizer, question, model_output, pixel_values=pixel_values
        )
        tokens = t.tensor(input_ids).cuda().unsqueeze(0)
        input_embeds = self.get_input_embeds(pixel_values, tokens)
        return self.generate(tokens=tokens, inputs_embeds=input_embeds, **kwargs)

    def get_input_embeds(self, pixel_values, input_ids, visual_features=None):
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.model.extract_feature(pixel_values)
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.model.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        return input_embeds

    def get_logits(self, pixel_values, tokens):
        with t.no_grad():
            instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            self.set_from_positions(instr_pos)
            input_embeds = self.get_input_embeds(pixel_values, tokens)
            logits = self.model.language_model(inputs_embeds=input_embeds).logits
            return logits

    def get_logits_from_text(self, user_input: str, model_output: Optional[str] = None, system_prompt: Optional[str] = None) -> t.Tensor:
        pdb.set_trace()  # todo
        tokens = tokenizer_internvl_chat(
            tokenizer=self.tokenizer, user_input=user_input, model_output=model_output, system_prompt=system_prompt
        )
        tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
        return self.get_logits(tokens)

    def get_last_activations(self, layer):
        return self.model.language_model.model.layers[layer].activations

    def set_add_activations(self, layer, activations):
        self.model.language_model.model.layers[layer].add(activations)

    def set_calc_dot_product_with(self, layer, vector):
        pdb.set_trace()  # todo
        self.model.language_model.model.layers[layer].calc_dot_product_with = vector

    def get_dot_products(self, layer):
        pdb.set_trace()  # todo
        return self.model.language_model.model.layers[layer].dot_products

    def reset_all(self):
        for layer in self.model.language_model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        pdb.set_trace()  # todo
        data = self.get_activation_data(decoded_activations, topk)[0]
        print(label, data)

    def decode_all_layers(
        self,
        tokens,
        topk=10,
        print_attn_mech=True,
        print_intermediate_res=True,
        print_mlp=True,
        print_block=True,
    ):
        pdb.set_trace()  # todo
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        for i, layer in enumerate(self.model.language_model.model.layers):
            print(f"Layer {i}: Decoded intermediate outputs")
            if print_attn_mech:
                self.print_decoded_activations(
                    layer.attn_out_unembedded, "Attention mechanism", topk=topk
                )
            if print_intermediate_res:
                self.print_decoded_activations(
                    layer.intermediate_resid_unembedded,
                    "Intermediate residual stream",
                    topk=topk,
                )
            if print_mlp:
                self.print_decoded_activations(
                    layer.mlp_out_unembedded, "MLP output", topk=topk
                )
            if print_block:
                self.print_decoded_activations(
                    layer.block_output_unembedded, "Block output", topk=topk
                )

    def plot_decoded_activations_for_layer(self, layer_number, tokens, topk=10):
        pdb.set_trace()  # todo
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        layer = self.model.language_model.model.layers[layer_number]

        data = {}
        data["Attention mechanism"] = self.get_activation_data(
            layer.attn_out_unembedded, topk
        )[1]
        data["Intermediate residual stream"] = self.get_activation_data(
            layer.intermediate_resid_unembedded, topk
        )[1]
        data["MLP output"] = self.get_activation_data(layer.mlp_out_unembedded, topk)[1]
        data["Block output"] = self.get_activation_data(
            layer.block_output_unembedded, topk
        )[1]

        # Plotting
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
        fig.suptitle(f"Layer {layer_number}: Decoded Intermediate Outputs", fontsize=21)

        for ax, (mechanism, values) in zip(axes.flatten(), data.items()):
            tokens, scores = zip(*values)
            ax.barh(tokens, scores, color="skyblue")
            ax.set_title(mechanism)
            ax.set_xlabel("Value")
            ax.set_ylabel("Token")

            # Set scientific notation for x-axis labels when numbers are small
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def get_activation_data(self, decoded_activations, topk=10):
        pdb.set_trace()  # todo
        softmaxed = t.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = t.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))
