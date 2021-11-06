"""Blender operator example with redo panel. Brings it up via F3 search menu easily.

This is based on `Scripting for Artists` series from blender youtube channel.
Use `Text > Register` to run on load.

References:
    - https://www.youtube.com/playlist?list=PLa1F2ddGya_8acrgoQr1fTeIuQtkSd6BW
"""

import bpy


class MeshOperatorMonkeyGrid(bpy.types.Operator):
    """Move an object with the mouse, example
    """
    bl_idname = "mesh.monkey_grid"
    bl_label = "Monkey Grid"
    bl_options = {'REGISTER', 'UNDO'}
    
    count_x: bpy.props.IntProperty(
        name="X",
        description="Num of monkey in X direction",
        default=3,
        min=1,
        max=10
    )

    count_y: bpy.props.IntProperty(
        name="Y",
        description="Num of monkey in X direction",
        default=5,
        min=1,
        max=10
    )
    
    size: bpy.props.FloatProperty(
        name="Size",
        description="Money size.",
        default=0.8,
        min=0,
        max=1    
    )
    
    @classmethod
    def poll(self, context):
        """Only show the operator on selected area type such as 3d viewport.
        """
        return context.area.type == 'VIEW_3D'
    
    def execute(self, context):
        """Executes code when the operator is run for example, F3 > OP Name.
        
        This is a blocking operation.
        """
        for idx in range(self.count_x * self.count_y):
            x = idx % self.count_x
            y = idx // self.count_y
            bpy.ops.mesh.primitive_monkey_add(
                size=self.size,
                location=(x,y,1)
            )
        return {'FINISHED'}


def register():
    bpy.utils.register_class(MeshOperatorMonkeyGrid)


def unregister():
    bpy.utils.unregister_class(MeshOperatorMonkeyGrid)


if __name__ == "__main__":
    register()

#    # test call
#    bpy.ops.object.modal_operator('INVOKE_DEFAULT')
