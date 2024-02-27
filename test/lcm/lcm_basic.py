from tmunan.imagine.sd_lcm.lcm import LCM

if __name__ == '__main__':
    lcm = LCM(txt2img_size='large')
    lcm.load()
    res = lcm.txt2img(
        prompt='bunny running around screaming at everybody',
        num_inference_steps=5,
        guidance_scale=0.5,
        height=768, width=768,
        seed=123,
        randomize_seed=False
    )
    res[0].save('test_image.png')
