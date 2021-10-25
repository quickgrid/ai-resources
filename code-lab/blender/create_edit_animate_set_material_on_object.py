#####################################################################################
## Author: ASIF AHMED
## Link: quickgrid.blogspot.com
## Code: Create and animate an object
## Version: Blender 2.82
#####################################################################################


import bpy
import bmesh




if __name__ == "__main__":
    
    #####################################################################################
    ## Clean the view port by removing all objects
    ## Warning to not use in your existing project
    #####################################################################################


    ## Deselect all and select the required faces
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete() 
    
    
    #####################################################################################
    ## Object Create and Edit
    #####################################################################################
    
    ## Create object and name 
    bpy.ops.mesh.primitive_cube_add(enter_editmode=False, location=(0, 0, 0))
    bpy.context.active_object.name = 'RobotBody'


    #bpy.context.scene.objects["Cube"].select_set(True)
    #bpy.context.object.name = "RobotBody"


    ## Move the object upwards
    bpy.ops.transform.translate(value=(0, 0, 3), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)



    ## Enter edit mode
    bpy.ops.object.editmode_toggle()


    ## Sub divide the cube
    bpy.ops.mesh.subdivide()
    bpy.ops.mesh.subdivide()


    ##
    obj = bpy.context.edit_object
    me = obj.data
    bm = bmesh.from_edit_mesh(me)


    # notice in Bmesh polygons are called faces
    bm.faces.ensure_lookup_table()


    ## Deselect all and select the required faces
    bpy.ops.mesh.select_all(action = 'DESELECT')


    ## Select alternate faces
    for i in range(0, len(bm.faces)):
        if i % 2 == 0:
            bm.faces[i].select = True  # select index 4



    ## Show the updates in the viewport
    bmesh.update_edit_mesh(me, True)



    ## Extrude selected faces
    bpy.ops.mesh.extrude_region_move(MESH_OT_extrude_region={"use_normal_flip":False, "mirror":False}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_target":'CLOSEST', "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "release_confirm":False, "use_accurate":False})


    ## Scaled the extruded part
    bpy.ops.transform.resize(value=(2, 2, 2), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)



    ## Exit edit mode
    bpy.ops.object.editmode_toggle()


    ## Create a plane for better rendering
    bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, location=(0, 0, 0))
    bpy.context.active_object.name = 'Ground'
    bpy.ops.transform.resize(value=(100, 100, 100), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)



    #####################################################################################
    ## Animation Part
    #####################################################################################    

    ## Get the object, Deselect all objects, Make the cube the active object, Select the cube
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 0), rotation=(0.927188, 0.010901, 2.65285))
    ob = bpy.context.scene.objects["Camera"]        
    bpy.ops.object.select_all(action='DESELECT') 
    bpy.context.view_layer.objects.active = ob     
    ob.select_set(True)                           
    

    ## Move and position camera
    bpy.context.object.location[0] = 15
    bpy.context.object.location[1] = 30
    bpy.context.object.location[2] = 30
    
    
    ## Setup spot light
    bpy.ops.object.light_add(type='SPOT', location=(1.5, 2, 1.5))
    bpy.ops.transform.translate(value=(0, 0, 13.5667), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, True), mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    bpy.ops.transform.resize(value=(1.57121, 1.57121, 1.57121), orient_type='GLOBAL', orient_matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)), orient_matrix_type='GLOBAL', mirror=True, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False)
    
    ## Set energy 
    bpy.context.object.data.energy = 1200

    



    ## Set start and end frames for time line
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 100


    ## Set positions for animation
    start_pos = (0, 0, 3)
    positions = [start_pos, (3, 4, 1), (3, 8, 3), (9, 1, 4), (0, 1, 0), start_pos]
    frame_number = 0
    
    
    ## Select the object though it should not be needed
    ## if there is only one object and is also active
    ob = bpy.data.objects["RobotBody"]
    
    
    ## For each position go to frame, move to position and apply keyframe
    for position in positions:
        bpy.context.scene.frame_set(frame_number)
        ob.location = position
        ob.keyframe_insert(data_path = "location", index = -1)
        frame_number += 20
        
        
        
    #####################################################################################
    ## Set Materials for object                                      
    #####################################################################################

    ## Create material for ground
    ob = bpy.context.scene.objects["Ground"]        
    bpy.ops.object.select_all(action='DESELECT') 
    bpy.context.view_layer.objects.active = ob     
    ob.select_set(True)   
    

    ## Create new material
    mat = bpy.data.materials.new(name="MaterialGround")
    ob.data.materials.append(mat)
    bpy.context.object.active_material.use_nodes = True
    
    
    ## Set material properties
    mat_name = bpy.data.materials[mat.name]
    mat_name_principled = mat_name.node_tree.nodes["Principled BSDF"]
    mat_name_principled.inputs[0].default_value = (0.8, 0.5, 0.3, 1)
    mat_name_principled.inputs[7].default_value = 0.1
    mat_name_principled.inputs[4].default_value = 1
    mat_name_principled.node_tree.nodes["Principled BSDF"].inputs[12].default_value = 1




    
     
