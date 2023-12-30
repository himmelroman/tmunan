import asyncio
import uuid
from copy import deepcopy
from pathlib import Path

from tmunan.api.context import context
from tmunan.api.pydantic_models import ImageSequence, ImageSequenceScript, Instructions


def gen_seq_prompt(prompts, prog: float):

    def calc_weight(p):
        weight_step = (p.end_weight - p.start_weight) / 100   # How much is one percent?
        return p.start_weight + (weight_step * prog)          # How progressed is this sequence?

    def format_prompt(text, weight):
        return text if weight == 1.0 else f'({text}){round(weight, 3)}'

    # build master prompt string
    return ', '.join(format_prompt(p.text, calc_weight(p)) for p in prompts)


def generate_image_sequence(seq: ImageSequence, config: Instructions, seq_id, parent_dir=None):

    # prepare dir
    seq_dir = f'{parent_dir if parent_dir else context.cache_dir}/seq_{seq_id}/'
    Path.mkdir(Path(seq_dir), exist_ok=True, parents=True)

    # set up event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # verify seed
    if not config.seed:
        config.seed = context.lcm.get_random_seed()

    # iterate as many times as requested
    for i in range(0, seq.num_images):

        # gen prompt for current sequence progress
        prompt = gen_seq_prompt(seq.prompts, (i / seq.num_images * 100))
        print(f'Generating image {i} with prompt: {prompt}')

        # gen image
        images = context.lcm.txt2img(
            prompt=prompt,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            height=config.height, width=config.width,
            seed=config.seed,
            randomize_seed=False
        )

        # save image to disk
        image_path = f'{seq_dir}/{i}'
        image_relative_path = str(Path(image_path).relative_to(Path(context.cache_dir)))
        images[0].save(f'{image_path}.png')

        # notify image ready
        if context.ws_manager.active_connections:
            loop.run_until_complete(
                context.ws_manager.active_connections[0].send_json({
                    'event': 'IMAGE_READY',
                    'sequence_id': seq_id,
                    'image_id': image_relative_path,
                    'prompt': prompt,
                    'seed': config.seed
                })
            )

    # notify sequence ended
    if context.ws_manager.active_connections:

        loop.run_until_complete(
            context.ws_manager.active_connections[0].send_json({
                'event': 'SEQUENCE_FINISHED',
                'sequence_id': seq_id
            })
        )


def generate_script(script: ImageSequenceScript, config: Instructions, script_id):

    # prepare dir
    script_dir = f'{context.cache_dir}/script_{script_id}/'
    Path.mkdir(Path(script_dir), exist_ok=True, parents=True)

    # iterate as many times as requested
    for i, seq in enumerate(script.sequences):

        # gen seq id
        seq_id = str(uuid.uuid4())[:8]

        # if this is NOT the first sequence
        if i > 0:

            # iterate prompts from previous sequence
            for p in script.sequences[i - 1].prompts:

                # reverse weight trajectories
                reversed_prompt = deepcopy(p)
                reversed_prompt.start_weight = reversed_prompt.end_weight
                reversed_prompt.end_weight = 0.0

                # add reversed prompt to new sequence
                seq.prompts.append(reversed_prompt)

        # play sequence
        generate_image_sequence(seq, config, seq_id=seq_id, parent_dir=script_dir)
