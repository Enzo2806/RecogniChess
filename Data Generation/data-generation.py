import numpy as np


# Method to print in the blender console
import bpy
def print(data):
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'CONSOLE':
                override = {'window': window, 'screen': screen, 'area': area}
                bpy.ops.console.scrollback_append(override, text=str(data), type="OUTPUT")    

# Store all chess pawn objects in separate variables
white_bishop_1 = bpy.data.objects["Bishop"]
white_bishop_2 = bpy.data.objects["Bishop.001"]
black_bishop_1 = bpy.data.objects["Bishop.002"]
black_bishop_2 = bpy.data.objects["Bishop.003"]

white_king = bpy.data.objects["King"]
black_king = bpy.data.objects["King.001"]

white_knight_1 = bpy.data.objects["Knight"]
white_knight_2 = bpy.data.objects["Knight.001"]
black_knight_1 = bpy.data.objects["Knight.002"]
black_knight_2 = bpy.data.objects["Knight.003"]

pawn_white_1 = bpy.data.objects["Pawn.001"]
pawn_white_2 = bpy.data.objects["Pawn.002"]
pawn_white_3 = bpy.data.objects["Pawn.003"]
pawn_white_4 = bpy.data.objects["Pawn.004"]
pawn_white_5 = bpy.data.objects["Pawn.005"]
pawn_white_6 = bpy.data.objects["Pawn.006"]
pawn_white_7 = bpy.data.objects["Pawn.007"]
pawn_white_8 = bpy.data.objects["Pawn.008"]

pawn_black_1 = bpy.data.objects["Pawn.009"]
pawn_black_2 = bpy.data.objects["Pawn.010"]
pawn_black_3 = bpy.data.objects["Pawn.011"]
pawn_black_4 = bpy.data.objects["Pawn.012"]
pawn_black_5 = bpy.data.objects["Pawn.013"]
pawn_black_6 = bpy.data.objects["Pawn.014"]
pawn_black_7 = bpy.data.objects["Pawn.015"]
pawn_black_8 = bpy.data.objects["Pawn.016"]

white_queen = bpy.data.objects["Queen"]
black_queen = bpy.data.objects["Queen.001"]

white_rook_1 = bpy.data.objects["Rook"]
white_rook_2 = bpy.data.objects["Rook.001"]
black_rook_1 = bpy.data.objects["Rook.002"]
black_rook_2 = bpy.data.objects["Rook.003"]

# Create an array storing all possible locations on the board
# We will use a dictionnary to store the center point for each poissible position
# We know the coordinate for the center of A1 is 
# X = 3.5 ; Y = 3.5 ; Z = 1.7842
# We therefore store it in the dictionnary as the main reference to store the 
# coordinates of the other locations
locations = {'A1': [3.5, 3.5, 1.7842]}

# We know moving on the X axis changes the column of the pawn, and each location center is separated by 
# exaclty 1 meter. Thus moving from A1 to B1 is done by removing 1 meter on the X coordinate.
# The same principle is applied to move on rows: moving from A1 to A2 is done by removing 1 meter on the Y coordinate.
# The Z coordinate remains constant.
column = 'A' # Start by storing coordinates for the A column

# We loop over all possible location and store the corresponding center of each coordinate 
# in the "locations" datastructure.
for i in range (8):
    for j in range (8):
        square = chr(ord(column) + i) + str(j+1) # Create the name of the location, ex: A2, B6 ...
        locations[square] = [3.5-i, 3.5-j, 1.7842] # Create a new entry in the dictionnary and store the cooresponding coordinates
        
# Test --> Check that the black rook on H8 and the coordinate we computed for H8 are the same
print("TEST: Check that the black rook on H8 and the coordinate we computed for H8 are the same")
print(locations["H8"])
print(black_rook_2.location)

# Variable to store the number of examples to generate
# The test set will have around 1/4*dataset_size examples in it 
# The training set will have 3/4*dataset_size examples in it
dataset_size = 10

for i in range (dataset_size):
    

