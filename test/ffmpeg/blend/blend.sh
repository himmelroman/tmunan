#ffmpeg -framerate 1/4 -pattern_type glob -i "*.png" -vf "minterpolate=fps=3:mi_mode=mci:mc_mode=obmc:me_mode=bidir:me=ds:vsbmc=0.5" -r 12 out.mp4
#ffmpeg -framerate 1/2 -pattern_type glob -i "*.png" -vf "minterpolate=fps=25:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=0.9" out.mp4
#ffmpeg -framerate 1/2 -pattern_type glob -i "*.png" -vf "minterpolate=fps=25:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=0.9, unsharp=luma_msize_x=7:luma_msize_y=7:luma_amount=2.5" out.mp4
ffmpeg -framerate 1/3 -pattern_type glob -i "*.png" -vf "minterpolate=fps=12:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=0.9, cas=0.9" out.mp4

# ffmpeg -hwaccel cuda -hwaccel_output_format cuda -pattern_type glob -i "*.png" -c:v h264_nvenc out.mp4
