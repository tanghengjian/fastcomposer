#CAPTION="a man <|image|> and a man <|image|> are reading book together"
#DEMO_NAME="newton_einstein"
#CAPTION="a man <|image|> and a man <|image|> and a man <|image|> are reading book together"
CAPTION="a man <|image|> and woman <|image|> standing next to each other in a room"
DEMO_NAME="man_woman"
#finetuned_model_path_name="model/fastcomposer"
#finetuned_model_path_name="models/stable-diffusion-v1-5/train_test"
#--pretrained_model_name_or_path model/runwayml/stable-diffusion-v1-5 \
#finetuned_model_path_name="models/YamerMIX_v8/train_test/postfuse-localize-ffhq-1_5-1e-5/checkpoint-15000"
finetuned_model_path_name="models/YamerMIX_v8/train_test/sdxl/checkpoint-400"
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --mixed_precision=fp16 \
    fastcomposer/inference.py \
    --pretrained_model_name_or_path model/wangqixun/YamerMIX_v8 \
    --finetuned_model_path ${finetuned_model_path_name} \
    --test_reference_folder data/${DEMO_NAME} \
    --test_caption "${CAPTION}" \
    --output_dir outputs/${DEMO_NAME} \
    --mixed_precision fp16 \
    --image_encoder_type clip \
    --image_encoder_name_or_path model/openai/clip-vit-large-patch14 \
    --num_image_tokens 1 \
    --max_num_objects 2 \
    --object_resolution 224 \
    --generate_height 512 \
    --generate_width 512 \
    --num_images_per_prompt 1 \
    --num_rows 1 \
    --seed 42 \
    --guidance_scale 5 \
    --inference_steps 50 \
    --start_merge_step 10 \
    --no_object_augmentation

