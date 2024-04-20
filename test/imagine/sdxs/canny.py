from PIL import ImageFilter
from diffusers.utils import load_image, make_image_grid

image = load_image('/Users/himmelroman/Desktop/Bialik/me.png')
g_image = image.convert("L")
edge_image = g_image.filter(ImageFilter.FIND_EDGES)
# edge_image.save('/Users/himmelroman/Desktop/Bialik/me_canny.png')

make_image_grid([image, g_image, edge_image], 1, 3).show()
