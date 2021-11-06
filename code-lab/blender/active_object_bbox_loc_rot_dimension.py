"""Get the location, rotation(radian) and dimension of selected object bounding box.

References
- https://blender.stackexchange.com/questions/14070/create-and-export-bounding-boxes-for-objects
"""

import bpy

selected = bpy.context.selected_objects

for obj in selected:
    #ensure origin is centered on bounding box center
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    #create a cube for the bounding box
    bpy.ops.mesh.primitive_cube_add() 
    #our new cube is now the active object, so we can keep track of it in a variable:
    bound_box = bpy.context.active_object 
    
    bpy.context.active_object.display_type = 'WIRE'

    #copy transforms
    bound_box.dimensions = obj.dimensions
    bound_box.location = obj.location
    bound_box.rotation_euler = obj.rotation_euler
    
    print(obj.dimensions)
    print(obj.location)
    print(obj.rotation_euler)
