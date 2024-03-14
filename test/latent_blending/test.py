import torch
import warnings

from compel import Compel, ReturnedEmbeddingsType

from diffusers import AutoPipelineForText2Image
from latentblending import add_frames_linear_interp
from latentblending.blending_engine import BlendingEngine


def get_text_embedding(self, prompt):

    print('My embeddings are running!')
    compel = Compel(
                    tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                    text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True]
                )

    # create compel prompt components
    conditioning, pooled = compel(prompt)

    do_classifier_free_guidance = self.guidance_scale > 1 and self.pipe.unet.config.time_cond_proj_dim is None
    text_embeddings = self.pipe.encode_prompt(
        # prompt=prompt,
        # prompt_2=prompt,
        device=self.pipe._execution_device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=self.negative_prompt,
        negative_prompt_2=self.negative_prompt,
        prompt_embeds=conditioning,
        pooled_prompt_embeds=pooled,
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        lora_scale=None,
        clip_skip=None,  # self.pipe._clip_skip,
    )
    return text_embeddings


warnings.filterwarnings('ignore')
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False

# %% First let us spawn a stable diffusion holder. Uncomment your version of choice.
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("mps")

be = BlendingEngine(pipe)
be.get_text_embeddings = get_text_embedding
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
