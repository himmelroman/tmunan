import time
import torch

from PIL import Image

from torch import nn
import torchvision.transforms as transforms


class ESRGAN(nn.Module):
    def __init__(self):
        super(ESRGAN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def benchmark(model, input_tensor, num_runs=100):
    total_time = 0
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = model(input_tensor)
        total_time += time.time() - start_time
    return total_time / num_runs


# Check if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model
model = ESRGAN().to(device)
# Note: You'll need to have pre-trained weights or train the model
# model.load_state_dict(torch.load('esrgan_weights.pth', map_location=device))
model.eval()

# Prepare the image
img = Image.open('/Users/himmelroman/projects/speechualizer/tmunan/test/upscale/diffusers/astronaut_512.png')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0).to(device)

# Run benchmark
latency = benchmark(model, input_tensor)
print(f"ESRGAN - Average latency: {latency * 1000:.2f} ms")
