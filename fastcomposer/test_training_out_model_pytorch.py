#from diffusers import DiffusionPipeline
from diffusers import StableDiffusionXLPipeline
import torch
#model/wangqixun/YamerMIX_v8
#models/YamerMIX_v8/train_test/sdxl
#models/YamerMIX_v8/train_test/sdxl/checkpoint-400/pytorch_model.bin
pipe = StableDiffusionXLPipeline.from_pretrained(f"models/YamerMIX_v8/train_test/sdxl/checkpoint-400/pytorch_model.bin", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "a man and woman standing next to each other in a room"

images = pipe(prompt=prompt).images[0]
images.save("usepytorch.jpg")

'''
from optimum.onnxruntime import ORTStableDiffusionXLPipeline

#model_id = "models/YamerMIX_v8/train_test/postfuse-localize-ffhq-1_5-1e-5/checkpoint-15000/pytorch_model.bin"
pipeline = ORTStableDiffusionXLPipeline.from_pretrained("models/YamerMIX_v8/train_test/postfuse-localize-ffhq-1_5-1e-5/checkpoint-15000")
pipeline.to("cuda")
prompt = "A majestic lion jumping from a big stone at night"
image = pipeline(prompt).images[0]
image.save("usepytorch.jpg")
'''