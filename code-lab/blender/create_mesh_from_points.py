"""
Generates a 2d face like mesh.

Reference,
https://blender.stackexchange.com/questions/67517/possible-to-add-a-plane-using-vertices-via-pythons
https://www.reddit.com/r/desmos/comments/dfnpe2/anyway_to_exportcopypaste_data_out_of_a_desmos/
"""

import bpy
import numpy as np

def create_custom_mesh(objname, px, py, pz):
    """Creates a mesh based on given points and location.
    
    Args:
        objname: Name of the object shown in scene collection.
        px: X coordinate of where to draw object.
        py: X coordinate of where to draw object.
        pz: X coordinate of where to draw object.
    
    Returns:
        A new bpy object with given args.   
    """
    
    
    # Define arrays for holding data    
    myvertex = []
    myfaces = []

    # Create all Vertices
    # Numpy flip can be used to reflect points along axis to reduce work of figuring out all points
    # Use a 2d plotting app such as desmos, geogebra to generate necessary points for shape
    x_loc = [2,-2,4,-4,4,-4,4,1,-1,-4]
    y_loc = [4,4,2,2,0,0,-2,-6,-6,-2]
    z_loc = np.zeros_like(x_loc)
    #z_loc = np.ones_like(x_loc)

    location_tensor = np.vstack((x_loc, y_loc, z_loc))
    print(location_tensor)

    for i in range(len(x_loc)):
        mypoint = [tuple(location_tensor[:, i])]
        print(mypoint)
        myvertex.extend(mypoint)        


    # -------------------------------------
    # Create all Faces
    # How to get point ordering?
    # Get the location index of points in couter clockwise order
    # Clockwise could also work
    # -------------------------------------
    myface = [(0, 1, 3, 5, 9, 8, 7, 6, 4, 2)]
    myfaces.extend(myface)

    
    mymesh = bpy.data.meshes.new(objname)

    myobject = bpy.data.objects.new(objname, mymesh)

    bpy.context.scene.collection.objects.link(myobject)
    
    # Generate mesh data
    mymesh.from_pydata(myvertex, [], myfaces)
    # Calculate the edges
    mymesh.update(calc_edges=True)

    # Set Location
    myobject.location.x = px
    myobject.location.y = py
    myobject.location.z = pz

    return myobject

curloc = bpy.context.scene.cursor.location
create_custom_mesh("Awesome_object", curloc[0], curloc[1], 0)