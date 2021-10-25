"""
Get rotation values in querternion, euler. 
Use keyframes to create animated object first.
Write the angle data to a text file.
"""

import bpy


scn = bpy.context.scene
obj = bpy.context.object


i = 0
with open('C:\\tmp\\data_out.txt', 'w') as out_file:
    for f in range(scn.frame_start, scn.frame_end):
    
        
        """
        ## Use below code to get values 
        # obj. =====> (location, rotation_euler, scale)
        # obj.matrix_world ====> (.to_quaternion(), .to_euler()) 
        """
        
        #print(obj.matrix_world.to_quaternion())
        #print(obj.matrix_world.to_euler(), obj.rotation_euler)
        #print(obj.rotation_euler)

        
        v1 = obj.rotation_euler[0]
        v2 = obj.rotation_euler[1]
        v3 = obj.rotation_euler[2]

        
        """
        output is printed to blender system console
        """        
        i+=1        
        print(v1, v2, v3, i)


        out_file.write('{},{:.3f},{:.3f},{:.3f}\n'.format(f,v1,v2,v3))
