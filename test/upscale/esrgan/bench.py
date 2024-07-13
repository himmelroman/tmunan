import time
import torch
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def benchmark(upsampler, input_tensor, num_runs=10):
    total_time = 0
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _, _ = upsampler.enhance(input_tensor.squeeze().permute(1, 2, 0).cpu().numpy())
        total_time += time.time() - start_time
    return total_time / num_runs


# Check if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model
model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=2, act_type='prelu')
model = model.to(device)
upsampler = RealESRGANer(scale=2, model_path='/Users/himmelroman/projects/speechualizer/tmunan/test/upscale/esrgan/realesr-general-x4v3.pth', model=model, tile=0, tile_pad=10, pre_pad=0)

# Read and prepare the image
img = Image.open('input.png').convert('RGB')
input_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

# Run benchmark
latency = benchmark(upsampler, input_tensor)
print(f"Real-ESRGAN - Average latency: {latency * 1000:.2f} ms")
