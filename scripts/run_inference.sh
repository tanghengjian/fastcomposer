#CAPTION="a man <|image|> and a man <|image|> are reading book together"
#DEMO_NAME="newton_einstein"
CAPTION="a man <|image|> and a man <|image|> and a man <|image|> are reading book together"
DEMO_NAME="3man"

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --mixed_precision=fp16 \
    fastcomposer/inference.py \
    --pretrained_model_name_or_path model/runwayml/stable-diffusion-v1-5 \
    --finetuned_model_path model/fastcomposer \
    --test_reference_folder data/${DEMO_NAME} \
    --test_caption "${CAPTION}" \
    --output_dir outputs/${DEMO_NAME} \
    --mixed_precision fp16 \
    --image_encoder_type clip \
    --image_encoder_name_or_path model/openai/clip-vit-large-patch14 \
    --num_image_tokens 1 \
    --max_num_objects 3 \
    --object_resolution 224 \
    --generate_height 960 \
    --generate_width 768 \
    --num_images_per_prompt 1 \
    --num_rows 1 \
    --seed 42 \
    --guidance_scale 5 \
    --inference_steps 50 \
    --start_merge_step 10 \
    --no_object_augmentation
