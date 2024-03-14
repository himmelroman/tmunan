import torch
import warnings
from tqdm import tqdm
from diffusers import AutoPipelineForText2Image
from latentblending import add_frames_linear_interp
from latentblending.blending_engine import BlendingEngine


warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False

# %% First let us spawn a stable diffusion holder. Uncomment your version of choice.
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("mps")

be = BlendingEngine(pipe)
be.set_prompt1("lightly cloudy sky background, (pink clouds)0.1")
be.set_prompt2("(lightly cloudy sky background)0.3, (pink clouds)1.0")
be.set_negative_prompt("blurry, ugly, pale")

# Run latent blending
be.run_transition()


def write_images(images, output_dir, duration_transition, fps=30):
    r"""
    Writes the transition movie to fp_movie, using the given duration and fps..
    The missing frames are linearly interpolated.
    Args:
        fp_movie: str
            file pointer to the final movie.
        duration_transition: float
            duration of the movie in seonds
        fps: int
            fps of the movie
    """

    # Let's get more cheap frames via linear interpolation (duration_transition*fps frames)
    imgs_transition_ext = add_frames_linear_interp(images, duration_transition, fps)

    # Save
    for i, img in enumerate(imgs_transition_ext):
        img = be.dh.latent2image(img)
        img.save(f'{output_dir}/{i}.png')


# Save movie
# be.write_movie_transition('movie_example1.mp4', duration_transition=12)
write_images(be.tree_final_imgs, '/tmp/blend_images', duration_transition=12)
