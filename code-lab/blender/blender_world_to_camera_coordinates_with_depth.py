"""Get world coordinates to image coordinates with depth.

Warning: The image coordinates are flipped, so they or the image must be flipped horizontally to match.

References:
    - https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex
"""

import bpy
from bpy_extras.object_utils import world_to_camera_view

scene = bpy.context.scene

# needed to rescale 2d coordinates
render = scene.render
res_x = render.resolution_x
res_y = render.resolution_y

obj = bpy.data.objects['Cube']
cam = bpy.data.objects['Camera']


# Get 2d image point only.
# use generator expressions () or list comprehensions []
verts = (vert.co for vert in obj.data.vertices)

for coord in verts:
    print(coord)

coords_2d = [world_to_camera_view(scene, cam, coord) for coord in verts]
print(coords_2d)

# 2d data printout:
rnd = lambda i: round(i)

print('x,y')
for x, y, distance_to_lens in coords_2d:
    print("{},{}".format(rnd(res_x*x), rnd(res_y*y)))

    

# Get 2d image point with depth.    
verts = (vert.co for vert in obj.data.vertices)
coords_2d = [world_to_camera_view(scene, cam, coord) for coord in verts]

# find min max distance, between eye and coordinate.
rnd = lambda i: round(i)
rnd3 = lambda i: round(i, 3)

limit_finder = lambda f: f(coords_2d, key=lambda i: i[2])[2]
limits = limit_finder(min), limit_finder(max)
limits = [rnd3(d) for d in limits]

print('min, max\n{},{}'.format(*limits))

# x, y, d=distance_to_lens
print('x,y,d')
for x, y, d in coords_2d:
    print("{},{},{}".format(rnd(res_x*x), rnd(res_y*y), rnd3(d)))