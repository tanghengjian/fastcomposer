from fastcomposer.transforms import get_object_transforms
from fastcomposer.data import DemoDataset
from fastcomposer.model import FastComposerModel
from diffusers import StableDiffusionPipeline,StableDiffusionXLPipeline
from transformers import CLIPTokenizer
from accelerate.utils import set_seed
from fastcomposer.utils import parse_args
from accelerate import Accelerator
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import os
from tqdm.auto import tqdm
from fastcomposer.pipeline import (
    stable_diffusion_call_with_references_delayed_conditioning,
)
import types
import itertools
import os

@torch.no_grad()
def main():
    args = parse_args()
    args.pretrained_model_name_or_path=f"model/wangqixun/YamerMIX_v8"
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype
    )

    model = FastComposerModel.from_pretrained(args)

    ckpt_name = "pytorch_model.bin"
    print(f"args.finetuned_model_path:{args.finetuned_model_path}")
    model.load_state_dict(
        torch.load(Path(args.finetuned_model_path) / ckpt_name, map_location="cpu")
    )

    model = model.to(device=accelerator.device, dtype=weight_dtype)

    pipe.unet = model.unet

    if args.enable_xformers_memory_efficient_attention:
        pipe.unet.enable_xformers_memory_efficient_attention()

    pipe.text_encoder = model.text_encoder
    pipe.image_encoder = model.image_encoder

    pipe.postfuse_module = model.postfuse_module

    pipe.inference = types.MethodType(
        stable_diffusion_call_with_references_delayed_conditioning, pipe
    )

    del model

    pipe = pipe.to(accelerator.device)

    # Set up the dataset
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    object_transforms = get_object_transforms(args)

    demo_dataset = DemoDataset(
        test_caption=args.test_caption,
        test_reference_folder=args.test_reference_folder,
        tokenizer=tokenizer,
        object_transforms=object_transforms,
        device=accelerator.device,
        max_num_objects=args.max_num_objects,
    )

    image_ids = os.listdir(args.test_reference_folder)
    print(f"Image IDs: {image_ids}")
    demo_dataset.set_image_ids(image_ids)

    unique_token = "<|image|>"

    prompt = args.test_caption
    prompt_text_only = prompt.replace(unique_token, "")

    os.makedirs(args.output_dir, exist_ok=True)

    batch = demo_dataset.get_data()

    input_ids = batch["input_ids"].to(accelerator.device)
    text = tokenizer.batch_decode(input_ids)[0]
    print(prompt)
    # print(input_ids)
    image_token_mask = batch["image_token_mask"].to(accelerator.device)

    # print(image_token_mask)
    all_object_pixel_values = (
        batch["object_pixel_values"].unsqueeze(0).to(accelerator.device)
    )
    num_objects = batch["num_objects"].unsqueeze(0).to(accelerator.device)

    all_object_pixel_values = all_object_pixel_values.to(
        dtype=weight_dtype, device=accelerator.device
    )

    object_pixel_values = all_object_pixel_values  # [:, 0, :, :, :]
    if pipe.image_encoder is not None:
        object_embeds = pipe.image_encoder(object_pixel_values)
    else:
        object_embeds = None

    encoder_hidden_states = pipe.text_encoder(
        input_ids, image_token_mask, object_embeds, num_objects
    )[0]
    


    print(f"test,prompt_text_only:{prompt_text_only}")
    encoder_hidden_states_text_only,_,pooled_text_embeds0,_ = pipe.encode_prompt(
        prompt_text_only,
        None,
        accelerator.device,
        args.num_images_per_prompt,
        do_classifier_free_guidance=False,
    )
    print(f"test,encoder_hidden_states_text_only.shape:{encoder_hidden_states_text_only.shape},pooled_text_embeds0.shape:{pooled_text_embeds0.shape}")


    text_input_ids_2 = pipe.tokenizer_2(
        prompt_text_only,
        max_length=pipe.tokenizer_2.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids

    encoder_output_2 = pipe.text_encoder_2(text_input_ids_2.to(accelerator.device), output_hidden_states=True)
    pooled_text_embeds = encoder_output_2[0]
    text_embeds_2 = encoder_output_2.hidden_states[-2]
    print(f"test,text_embeds_2.shape:{text_embeds_2.shape},pooled_text_embeds.shape:{pooled_text_embeds.shape}")


    encoder_hidden_states = pipe.postfuse_module(
        encoder_hidden_states,
        object_embeds,
        image_token_mask,
        num_objects,
    )
    print(f"test,00,encoder_hidden_states.shape:{encoder_hidden_states.shape}")


    #encoder_hidden_states = torch.concat([encoder_hidden_states, text_embeds_2], dim=-1) # concat

    print(f"test,11,encoder_hidden_states.shape:{encoder_hidden_states.shape}")

    cross_attention_kwargs = {}
    
    #prompt_embeds 携带了image信息
    #prompt_embeds_text_only只有原始文本信息
    images = pipe.inference(
        #prompt=prompt_text_only,
        prompt_embeds=encoder_hidden_states,
        num_inference_steps=args.inference_steps,
        height=args.generate_height,
        width=args.generate_width,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images_per_prompt,
        cross_attention_kwargs=cross_attention_kwargs,
        prompt_embeds_text_only=encoder_hidden_states_text_only,
        start_merge_step=args.start_merge_step,
        pooled_prompt_embeds = pooled_text_embeds,
        text_embeds_2 = text_embeds_2,
    ).images

    for instance_id in range(args.num_images_per_prompt):
        images[instance_id].save(
            os.path.join(
                args.output_dir,
                f"output_{instance_id}.png",
            )
        )


if __name__ == "__main__":
    main()
