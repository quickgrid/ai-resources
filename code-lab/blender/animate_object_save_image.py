"""
1. Replace `Cube` with the object name to animate that object in various angles and render.
2. This script can be used to generate view of object in various angles for angle regression, multi view, synthetic dataset generation etc.

References, 
https://blender.stackexchange.com/questions/73697/render-animation-with-different-angles
https://blender.stackexchange.com/questions/67517/possible-to-add-a-plane-using-vertices-via-python
"""

import bpy
from math import radians
from os.path import join

S = bpy.context.scene

renderFolder = "C:/tmp/tmp"

#camParent = bpy.data.objects['Empty']
#bpy.context.scene.objects["Cube"].select_set(True)
camParent = bpy.data.objects['Cube']

animLen   = 1 # num frames per angle
numAngles = 30
rotAngle  = 360 / numAngles

for i in range(numAngles):
    # Set camera angle via parent
    angle = i * rotAngle
    camParent.rotation_euler.z = radians( angle )

    # Render animation
    for f in range(1,animLen + 1):
        S.frame_set( f ) # Set frame

        frmNum   = str( f ).zfill(3) # Formats 5 --> 005
        fileName = "angle_{a}_frm_{f}".format( a = angle, f = frmNum )
        fileName += S.render.file_extension
        bpy.context.scene.render.filepath = join( renderFolder, fileName )

        bpy.ops.render.render(write_still = True)
