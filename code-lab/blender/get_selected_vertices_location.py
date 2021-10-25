"""
Task: Get the indices of selected vertices

To get indices of the object enable Blender Addon MesaureIt, go right sidebar(N key) on 3d viewport
and select Vertices button on Mesh Debug option.

Alternate way in Blender 2.8+ is to tick Developer Extras option on Preferences > Developer Option
and tick Developer > Indices on Overlays button on 3d viewport.
"""


import bpy
import bmesh


obj=bpy.context.object



############################################
## Geet location of selected points
## Link: https://stackoverflow.com/questions/15429796/blender-scripting-indices-of-selected-vertices
############################################
if obj.mode == 'EDIT':
    bm=bmesh.from_edit_mesh(obj.data)
    print("\n\n\n")
    for v in bm.verts:
        print(v.index)
        if v.select:
            print(v.co)
else:
    print("Object is not in edit mode.")
    


############################################
## Get location of custom points
############################################
custom_vertex_list = [1,5,6,7]
    
if obj.mode == 'EDIT':
    bm=bmesh.from_edit_mesh(obj.data)
    print("\n\n\n")
    for i in custom_vertex_list:
        v = bm.verts[i]
        print(v.co)
else:
    print("Object is not in edit mode.")
    
    
