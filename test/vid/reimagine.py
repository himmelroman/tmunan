import cv2
import torch
import numpy as np
from PIL import Image

from diffusers.utils import make_image_grid
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

from tmunan.imagine.sd_lcm.lcm import LCM
from tmunan.imagine.image_generator import ImageGeneratorRemote


def cv2_frame_to_pil_image(frame):
	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	return Image.fromarray(frame_rgb)


def pil_image_to_cv2_frame(img: Image):
	frame_rgb = np.asarray(img)
	return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def reimagine(image: Image, height, width):

	small_image = image.resize((height, width))

	res = igr.img2img_upload(
		image=small_image,
		prompt='black and white, (scratchboard)1.2',
		width=height,
		height=width,
		guidance_scale=0.8,
		num_inference_steps=4,
		strength=0.5,
		randomize_seed=True
	)
	return res


if __name__ == "__main__":

	device = "mps"
	weight_type = torch.float16

	# controlnet = ControlNetModel.from_pretrained(
	# 	"IDKiro/sdxs-512-dreamshaper-sketch", torch_dtype=weight_type
	# ).to(device)
	# pipe = StableDiffusionControlNetPipeline.from_pretrained(
	# 	"IDKiro/sdxs-512-dreamshaper", controlnet=controlnet, torch_dtype=weight_type
	# )
	# pipe.to(device)

	# lcm = LCM(model_size='small')
	# lcm.load()

	igr = ImageGeneratorRemote('http://3.255.31.250', '8080')

	# input video
	input_video = cv2.VideoCapture('/Users/himmelroman/Desktop/C1137.MP4')
	frames_per_second = int(input_video.get(cv2.CAP_PROP_FPS))
	frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

	# output video
	codec = cv2.VideoWriter.fourcc(*'mp4v')
	video_writer = cv2.VideoWriter('/Users/himmelroman/Desktop/C1137_bwscratch.MP4', codec, 1, (512, 512))

	# loop
	sample_rate = 50
	for fno in range(0, total_frames, sample_rate):

		print(f'Processing frame: {fno}')

		input_video.set(cv2.CAP_PROP_POS_FRAMES, fno)
		_, source_frame = input_video.read()

		# create reimagined frame
		pil_image = cv2_frame_to_pil_image(source_frame)
		reimage = reimagine(pil_image, 512, 512)
		new_frame = pil_image_to_cv2_frame(reimage)

		# dump
		grid = make_image_grid([pil_image, reimage], cols=2, rows=1)
		grid.save(f'/tmp/reimagine/reimage_{fno}.png')
		# pil_image.save(f'/tmp/reimagine/source_{fno}.png')

		# write the frame to the output video
		video_writer.write(new_frame)

	input_video.release()
	video_writer.release()
	cv2.destroyAllWindows()
