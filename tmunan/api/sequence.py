import asyncio

from tmunan.api.context import context
from tmunan.api.pydantic_models import ImageSequence


def gen_seq_prompt(prompts, prog: float):

    def calc_weight(p):
        weight_step = (p.max_weight - p.min_weight) / 100   # How much is one percent?
        return p.min_weight + (weight_step * prog)          # How progressed is this sequence?

    def format_prompt(text, weight):
        return text if weight == 1.0 else f'({text}){round(weight, 3)}'

    # build master prompt string
    return ', '.join(format_prompt(p.text, calc_weight(p)) for p in prompts)


def generate_image_sequence(seq: ImageSequence):

    # set up event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # verify seed
    if not seq.config.seed:
        seq.config.seed = context.lcm.get_random_seed()

    # iterate as many times as requested
    for i in range(0, seq.num_images):

        # gen prompt for current sequence progress
        prompt = gen_seq_prompt(seq.prompts, (i / seq.num_images * 100))
        print(f'Generating image {i} with prompt: {prompt}')

        # gen image
        images = context.lcm.txt2img(
            prompt=prompt,
            num_inference_steps=seq.config.num_inference_steps,
            guidance_scale=seq.config.guidance_scale,
            height=seq.config.height, width=seq.config.width,
            seed=seq.config.seed,
            randomize_seed=False
        )

        # # save image to disk
        image_id = f'img_seq_{i}'
        images[0].save(f'{context.cache_dir}/{image_id}.png')

        # notify image ready
        if context.ws_manager.active_connections:
            loop.run_until_complete(
                context.ws_manager.active_connections[0].send_json({
                    'event': 'IMAGE_READY',
                    'sequence_id': 1,
                    'image_id': image_id,
                    'prompt': prompt,
                    'seed': seq.config.seed
                })
            )

    # notify sequence ended
    if context.ws_manager.active_connections:

        loop.run_until_complete(
            context.ws_manager.active_connections[0].send_json({
                'event': 'SEQUENCE_FINISHED',
                'sequence_id': 1
            })
        )
