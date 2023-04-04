from argparse import ArgumentParser
from os import makedirs
import numpy as np
from pathlib import Path

import torch
import configparser
from transformers import PretrainedConfig

torch.set_printoptions(linewidth=130, sci_mode=False)
np.set_printoptions(linewidth=130, suppress=True)

# This converter is used to convert the huggingface gpt-j-6B model
# in https://huggingface.co/EleutherAI/gpt-j-6B/blob/main/pytorch_model.bin.

def savebin(param, save_path):
    print(f" enter savebin param : {param} , save_path : {save_path}")
    if isinstance(param, torch.Tensor):
        param = param.cpu().float().numpy()
        print(f" param from param.cpu().float().numpy() :  {param}")
    print(f" before np.squeeze ........")
    np.squeeze(param).astype(np.float32).tofile(save_path + ".bin")


def param2file(pt_param, layer_id, save_dir, dest_key):
    print(f" enter param2file pt_param : {pt_param} , layer_id : {layer_id} , save_dir : {save_dir}, dest_key : {dest_key}")
    base_n = save_dir + "/model.layers." + str(layer_id) + "."
    save_path = base_n + dest_key
    print(f" param2file base_n : {base_n} , save_path : {save_path} ")
    savebin(pt_param, save_path)

def param2distributed(
    pt_param,
    layer_id,
    save_dir,
    dest_key,
    n_inference_gpus,
    split_axis,
):
    np_param = pt_param.cpu().float().numpy()
    print(f" param2distributed  np_param : {np_param} , layer_id : {layer_id} ")
    print(f" param2distributed  save_dir : {save_dir} , dest_key : {dest_key} ")
    print(f" param2distributed  n_inference_gpus : {n_inference_gpus} , split_axis : {split_axis} ")
    base_n = save_dir + "/model.layers." + str(layer_id) + "."
    save_path = base_n + dest_key
    split_param = np.split(np_param, n_inference_gpus, axis=split_axis)
    print(f"   base_n : {base_n} , save_path : {save_path} ")
    print(f" split_param  :  {split_param} ")
    for i, p in enumerate(split_param):
        savebin(p, save_path + f".{i}")


def save(w, save_dir, n_inference_gpus, n_layers, layer_id):
    makedirs(save_dir, exist_ok=True)

    savebin(w['transformer.wte.weight'], save_dir + "/model.wte")
    l = layer_id
    print(f"Saving layer {l + 1} / {n_layers}")
    base_k = "transformer.h." + str(l) + "."
    param2file(
        w[base_k + "ln_1.bias"],
        l, save_dir, "input_layernorm.bias"
    )
    param2file(
        w[base_k + "ln_1.weight"],
        l, save_dir, "input_layernorm.weight"
    )
    param2distributed(
        w[base_k + "mlp.fc_in.weight"].T,
        l, save_dir, "mlp.dense_h_to_4h.weight",
        n_inference_gpus, split_axis=-1 # split fast indx
    )
    param2distributed(
        w[base_k + "mlp.fc_in.bias"],
        l, save_dir, "mlp.dense_h_to_4h.bias",
        n_inference_gpus, split_axis=-1 # split fast indx
    )

    param2distributed(
        w[base_k + "mlp.fc_out.weight"].T,
        l, save_dir, "mlp.dense_4h_to_h.weight",
        n_inference_gpus, split_axis=0  # split slow indx
    )
    param2file(
        w[base_k + "mlp.fc_out.bias"],
        l, save_dir, "mlp.dense_4h_to_h.bias"
    )
    param2distributed(
        w[base_k + "attn.out_proj.weight"].T,
        l, save_dir, "attention.dense.weight",
        n_inference_gpus, split_axis=0  # split slow indx
    )
    QKV_w = torch.stack([
        w[base_k + "attn.q_proj.weight"],
        w[base_k + "attn.k_proj.weight"],
        w[base_k + "attn.v_proj.weight"],
    ]) # [qkv, n_heads * dim_head, latent_space]
    QKV_w = QKV_w.permute(2, 0, 1)
    param2distributed(
        QKV_w, l, save_dir, "attention.query_key_value.weight",
        n_inference_gpus, split_axis=-1 # split fast indx
    )
    # Other unneeded per-layer params:
    # attn.attention.masked_bias = torch.tensor(-1e9)
    # attn.attention.bias = torch.tril(torch.ones(1, 1, 2048, 2048))

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Convert GPT-J slim checkpoint to FasterTransformer",
    )
    parser.add_argument(
        "--output-dir", help="Folder where binary files are stored", default="gpt-j-6B/c-models/"
    )
    parser.add_argument(
        "--ckpt-dir", help="File of GPT-J huggingface checkpoint", default="gpt-j-6B/"
    )
    parser.add_argument(
        "--n-inference-gpus", help="Number of GPUs used for inference runtime", default=1, type=int
    )
    parser.add_argument(
        "--n-layers", help="Number of GPT-J decoder layer", default=28, type=int
    )
    args = parser.parse_args()

    ckpt_file = args.ckpt_dir + "/pytorch_model.bin"
    checkpoint = torch.load(ckpt_file)
    print(f"loading from {ckpt_file}")

    out_path = args.output_dir
    output_dir = out_path + f"/{args.n_inference_gpus}-gpu/"
    print(f"saving to {output_dir}")

    config_file = args.ckpt_dir + "/config.json"
    print(f" load config_file : {config_file}")
    hf_config = PretrainedConfig.from_json_file(config_file).to_dict()

    print(f" load hf_config : {hf_config}")

    # NOTE: save parameters to config files (loaded by triton backends)
    config = configparser.ConfigParser()
    print(f" save parameters to config files (loaded by triton backends): {config}")

    config["gptj"] = {}
    try:
        config["gptj"]["model_name"] = "gptj" if hf_config["_name_or_path"] == '' else hf_config["_name_or_path"]
        gptj_model_name = config["gptj"]["model_name"]
        print(f" load gptj_model_name : {gptj_model_name}")
        config["gptj"]["head_num"] = str(hf_config["n_head"])
        gptj_head_num = config["gptj"]["head_num"]
        print(f" gptj_head_num : {gptj_model_name}")
        n_embd = hf_config["n_embd"]
        print(f" n_embd : {n_embd}")
        config["gptj"]["size_per_head"] = str(n_embd // hf_config["n_head"])
        gptj_size_per_head = config["gptj"]["size_per_head"]
        print(f" gptj size_per_head : {gptj_size_per_head}")
        config["gptj"]["inter_size"] = str(n_embd * 4)
        gptj_inter_size = config["gptj"]["inter_size"]
        print(f" gptj gptj_inter_size : {gptj_inter_size}")
        config["gptj"]["num_layer"] = str(hf_config["n_layer"])
        rotary_dim = n_embd // hf_config["n_head"] if hf_config["rotary_dim"] is None else hf_config["rotary_dim"]
        config["gptj"]["rotary_embedding"] = str(hf_config["rotary_dim"])
        config["gptj"]["vocab_size"] = str(hf_config["vocab_size"])
        config["gptj"]["start_id"] = str(hf_config["bos_token_id"])
        config["gptj"]["end_id"] = str(hf_config["eos_token_id"])
        config["gptj"]["weight_data_type"] = "fp32"
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        print(f" before write to {output_dir} /config.ini")
        with open(output_dir + "/config.ini", 'w') as configfile:
            config.write(configfile)
        print(f" after write to {output_dir} /config.ini")

    except:
        print(f"Fail to save the config in config.ini.")

    print(f" before for 28 iter save()")
    for i in range(args.n_layers):
        save(checkpoint, output_dir, args.n_inference_gpus, args.n_layers, i)
    
    print(f" savebin(checkpoint['transformer.ln_f.weight'] {output_dir} /model.final_layernorm.weight")
    savebin(checkpoint['transformer.ln_f.weight'], output_dir + "/model.final_layernorm.weight")
    print(f" savebin(checkpoint['transformer.ln_f.bias'] {output_dir} /model.final_layernorm.bias")
    savebin(checkpoint['transformer.ln_f.bias'], output_dir + "/model.final_layernorm.bias")
    print(f" savebin(checkpoint['lm_head.weight'] {output_dir} /model.lm_head.weight ")
    savebin(checkpoint['lm_head.weight'], output_dir + "/model.lm_head.weight")
    print(f" savebin(checkpoint['lm_head.bias'] {output_dir} /model.lm_head.bias")
    savebin(checkpoint['lm_head.bias'], output_dir + "/model.lm_head.bias")

    print("done")
