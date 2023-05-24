import random
import numpy as np
import os
from mathutils import Euler # to set rotation of pieces

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

# Light objects
light1 = bpy.data.objects["Spot"]
light2 = bpy.data.objects["Spot.001"]
light3 = bpy.data.objects["Spot.002"]
light4 = bpy.data.objects["Spot.003"]

# Create an array storing all possible locations on the board
# We will use a dictionnary to store the center point for each poissible position
# We know the coordinate for the center of A1 is 
# X = 3.5 ; Y = 3.5 ; Z = 1.7842
# We therefore store it in the dictionnary as the main reference to store the 
# coordinates of the other locations
locations = {'A1': (3.5, 3.5, 1.7842)}

# We know moving on the X axis changes the column of the chess piece, and each location center is separated by 
# exaclty 1 meter. Thus moving from A1 to B1 is done by removing 1 meter on the X coordinate.
# The same principle is applied to move on rows: moving from A1 to A2 is done by removing 1 meter on the Y coordinate.
# The Z coordinate remains constant.
column = 'A' # Start by storing coordinates for the A column

# We loop over all possible location and store the corresponding center of each coordinate 
# in the "locations" datastructure.
for i in range (8):
    for j in range (8):
        square = chr(ord(column) + i) + str(j+1) # Create the name of the location, ex: A2, B6 ...
        locations[square] = (3.5-i, 3.5-j, 1.7842) # Create a new entry in the dictionnary and store the cooresponding coordinates

# Create another location dictionnary for the possible ranodm locations of the pawns
# We consider that pawns cannot be placed at both ends of the board.
# Indeed they would be changed immediatly to a queen, rook, knight or bishop if they are on the other color's side
# And they can't move backwards. 
pawn_locations = locations.copy() # Shallow copy

# Iterate through all items in the available locations
for location in list(pawn_locations):
    # If they are the ends of the board i.e the location names ending with 0 or 8 (A0, B8...) 
    if location.endswith('1') or location.endswith('8'):
        # Delete it from the possible locations of pawns
        del pawn_locations[location]

# We also know that bishops that start on a color have to stay on it for the rest of the game
# We therefore also need to store all locations of white squares
# And all locations of green ones.         
all_white_squares = locations.copy()
all_green_squares = locations.copy()

# Iterate through all possible squares in the locations dictionnary
for location in list(locations):
    # Store unicode of row and column number
    row = ord(location[1])
    column = ord(location[0])
    
    # Unicode of numbers that are divisible by 2 are also divisble by 2.
    # Unicode for 'A' is 65 (so not divisble by 2).
    
    # If the row is not divisible by two but the column is --> B1, D1, H7... then the square is white
    if row % 2 == 1 and column % 2 == 0:
         del all_green_squares[location]
    # Else if the row is divisble by 2 but the column isn't --> A2, C2, G8.. then the square is white
    if row % 2 == 0 and column % 2 == 1:
         del all_green_squares[location]
    # Else if both the row and the column are not divisble by 2 --> A1, C1, E7.. then the square is green
    if row % 2 == 1 and column % 2 == 1:
         del all_white_squares[location]
    # Else if both the row and the column are divisble by 2 --> B2, D2, H8.. then the square is green
    if row % 2 == 0 and column % 2 == 0:
         del all_white_squares[location]

# Create an array containing all chess pieces
all_pieces = [white_bishop_1, white_bishop_2, white_king, white_knight_1, white_knight_2, white_queen,
pawn_white_1, pawn_white_2, pawn_white_3, pawn_white_4, pawn_white_5, pawn_white_6, pawn_white_7, pawn_white_8,
white_rook_1, white_rook_2, black_bishop_1, black_bishop_2, black_king, black_knight_1, black_knight_2, black_queen,
pawn_black_1, pawn_black_2, pawn_black_3, pawn_black_4, pawn_black_5, pawn_black_6, pawn_black_7, pawn_black_8,
black_rook_1, black_rook_2]

# Create an array containing all the white pawns
white_pieces = [white_bishop_1, white_bishop_2, white_king, white_knight_1, white_knight_2, white_queen,
pawn_white_1, pawn_white_2, pawn_white_3, pawn_white_4, pawn_white_5, pawn_white_6, pawn_white_7, pawn_white_8,
white_rook_1, white_rook_2]

# Create an array containing all the black pawns
black_pieces = [black_bishop_1, black_bishop_2, black_king, black_knight_1, black_knight_2, black_queen,
pawn_black_1, pawn_black_2, pawn_black_3, pawn_black_4, pawn_black_5, pawn_black_6, pawn_black_7, pawn_black_8,
black_rook_1, black_rook_2]

# Arrays containing all small pawns (black and white)
all_pawns = [pawn_black_1, pawn_black_2, pawn_black_3, pawn_black_4, pawn_black_5, pawn_black_6, pawn_black_7, pawn_black_8,
pawn_white_1, pawn_white_2, pawn_white_3, pawn_white_4, pawn_white_5, pawn_white_6, pawn_white_7, pawn_white_8]

# Location to put pieces we didn't randomly select
not_used_location = (6, 6, 2)

# Method to select pieces
def selectPieces():
    # Randomly select white and black pieces (random number of each)
    ran_white = random.sample(white_pieces, random.randint(0, len(white_pieces)))
    ran_black = random.sample(black_pieces, random.randint(0, len(black_pieces)))
    
    # Make sure both still have the king, otherwise the game would be over
    if white_king not in ran_white: ran_white.append(white_king)
    if black_king not in ran_black: ran_black.append(black_king)
    
     #return concatenated arrays 
    return ran_white + ran_black

# Method to retrieve pawn's name from object
def getName (piece):
    if piece == white_bishop_1 or piece == white_bishop_2:
        return "White Bishop"
    if piece == black_bishop_1 or piece == black_bishop_2:
        return "Black Bishop"
    if piece == white_king:
        return "White King"
    if piece == black_king:
        return "Black King"
    if piece == white_knight_1 or piece == white_knight_2:
        return "White Knight"
    if piece == black_knight_1 or piece == black_knight_2:
        return "Black Knight" 
    if piece in all_pawns and piece in white_pieces:
        return "White Pawn"
    if piece in all_pawns and piece in black_pieces:
        return "Black Pawn"
    if piece == white_queen:
        return "White Queen"
    if piece == black_queen:
        return "Black Queen"
    if piece == white_rook_1 or piece == white_rook_2:
        return "White Rook"
    if piece == black_rook_1 or piece == black_rook_2:
        return "Black Rook" 
    
# Method to assign random location to a pawn
def assignLocation(piece, label):
    # No matter the piece, we want to add some random rotation
    # Indeed chess pieces are rarely all aligned in the same direction.
    # However, since we added the knight and bishop pawns from another set.
    # its rotation properties are different (the Z axis is not the one we need to modify but the Y axis)
    # Thus we check if the piece is a knight or a bishop or not and modify its rotation accordingly
    if piece != white_knight_1 and piece != white_knight_2 and piece != black_knight_1 and piece != black_knight_2 :
        piece.rotation_euler = Euler((0, 0, random.uniform(0, 2*np.pi)), 'XYZ')
    else:
        piece.rotation_euler = Euler((0, random.uniform(0, 2*np.pi), 0), 'XYZ')
    
    # If the piece is a bishop, we always shift the coordinate X by pi/4 = 90 degrees and set the  Z axis to random
    if  piece == white_bishop_1 or piece == white_bishop_2 or piece == black_bishop_1 or piece == black_bishop_2:
        piece.rotation_euler = Euler((np.pi/2, 0, random.uniform(0, 2*np.pi)), 'XYZ')
    
    # Check if piece is pawn
    if piece in all_pawns:
        # get the new location, change it if it is already occupied by a piece
        while True: 
            # Select random location from available locations we computed above
            square_name, new_loc = random.choice(list(pawn_locations.items()))
            
            # If the square was not occupied break, otherwise find new location
            index = np.where(label == square_name)
            if label[index[0][0]][1] == '':
                break
            
        X = new_loc[0] + random.uniform(-0.10, 0.10)
        Y = new_loc[1] + random.uniform(-0.10, 0.10)
        Z = new_loc[2]
        
        piece.location = (X, Y, Z) # Assign new location
        
        # Store the placed pawn and its location in label array
        label[index[0][0]][1] = getName(piece)
    
        return label
        
    # Check if piece is a bishop starting on a white square
    if piece == white_bishop_1 or piece == black_bishop_1:
        # If it is, this piece must remain on a white square so we select a random
        # white square on the board and make it it's new location
        
        # get the new location, change it if it is already occupied by a piece
        while True:
            # Select random location from available locations we computed above
            square_name, new_loc = random.choice(list(all_white_squares.items()))
            
            # If the square was not occupied break, otherwise find new location
            index = np.where(label == square_name)
            if label[index[0][0]][1] == '':
                break
            
        X = new_loc[0] + random.uniform(-0.10, 0.10)
        Y = new_loc[1] + random.uniform(-0.10, 0.10)
        Z = 1.98816 # Bishops need to be placed a bit higher because of the center of origin being different (imported object)
        
        piece.location = (X, Y, Z) # Assign new location
        
        # Store the placed pawn and its location in label array
        label[index[0][0]][1] = getName(piece)
        
        return label
    
    # Check if piece is a bishop starting on a green square
    if piece == white_bishop_2 or piece == black_bishop_2:
        # If it is, this pawn must remain on a green square so we select a random
        # green square on the board and make it it's new location
        
        # get the new location, change it if it is already occupied by a piece
        while True:
            # Select random location from available locations we computed above
            square_name, new_loc = random.choice(list(all_green_squares.items()))
            
            # If the square was not occupied break, otherwise find new location
            index = np.where(label == square_name)
            if label[index[0][0]][1] == '':
                break
            
        X = new_loc[0] + random.uniform(-0.1, 0.1)
        Y = new_loc[1] + random.uniform(-0.1, 0.1)
        Z = 1.98816 # Bishops need to be placed a bit higher because of the center of origin being different (imported object)
        
        piece.location = (X, Y, Z) # Assign new location
        
        # Store the placed pawn and its location in label array
        label[index[0][0]][1] = getName(piece)
        
        return label
    
    # Place the black king
    if piece == black_king:
        # remember, we always place the white king before the black one 
        # because of the order in which we select them
        # So we place the black king in any location except the
        # locations around the white one.
        
        # Get the location of the white king
        white_king_location = label[np.where(label=="White King")[0]][0][0]
        
        # Get the X and Y coordinate of the white king
        white_king_X = locations[white_king_location][0]
        white_king_Y = locations[white_king_location][1]
        
        # Copy all locations into an aray that will store the possible locations for the black king
        black_king_locations = locations.copy()
        # Iterate through all possible squares in the locations dictionnary
        # Remove the locations next to the white king 
        for key, values in locations.items():
            X = values[0]
            Y = values[1]
            
            if X == white_king_X and (Y == white_king_Y or Y == white_king_Y+1 or Y == white_king_Y-1):
                del black_king_locations[key]
            if X == white_king_X+1 and (Y == white_king_Y or Y == white_king_Y+1 or Y == white_king_Y-1):
                del black_king_locations[key]
            if X == white_king_X-1 and (Y == white_king_Y or Y == white_king_Y+1 or Y == white_king_Y-1):
                del black_king_locations[key]
                
        # get the new location, change it if it is already occupied by a piece
        while True:
            # Select random location from available locations we computed above
            square_name, new_loc = random.choice(list(black_king_locations.items()))
            
            # If the square was not occupied break, otherwise find new location
            index = np.where(label == square_name)
            if label[index[0][0]][1] == '':
                break
        
        X = new_loc[0] + random.uniform(-0.10, 0.10)
        Y = new_loc[1] + random.uniform(-0.10, 0.10)
        Z = new_loc[2]
        
        piece.location = (X, Y, Z) # Assign new location
        
        # Store the placed pawn and its location in label array
        label[index[0][0]][1] = getName(piece)
        return label
        
    # Otherwise the piece can go anywhere on the board, so we just select a random location and assign it.
    else:
        # get the new location, change it if it is already occupied by a piece
        while True:
            # Select random location from available locations we computed above
            square_name, new_loc = random.choice(list(locations.items()))
            
            # If the square was not occupied break, otherwise find new location
            index = np.where(label == square_name)
            if label[index[0][0]][1] == '':
                break
            
        X = new_loc[0] + random.uniform(-0.10, 0.10)
        Y = new_loc[1] + random.uniform(-0.10, 0.10)
        Z = new_loc[2]
        
        piece.location = (X, Y, Z) # Assign new location
        
        # Store the placed pawn and its location in label array
        label[index[0][0]][1] = getName(piece)
        return label


# Variable to store the number of examples to generate
# Since we use this dataset for domain adaptation, all the examples will be in a training dataset
# All the examples will have to be labelled.
dataset_size = 4501

for i in range (dataset_size):
    
    # Array to store labels for this example
    label = ["Square number", "Chess Piece"]
    for square_name in list(locations.keys()):
        label = np.vstack([label, [square_name, '']]) # Populate label array for this example   
    
    # Randomly Select pieces
    selected_pieces =  selectPieces()
    
    # For each piece, find it a random location (some have specific rules see method above
    for piece in all_pieces:
        # Place the pieces that were randomly selected 
        if piece in selected_pieces: 
            # Place the piece by assignining it a location
            # Keep track of the bael array        
            label = assignLocation(piece, label)
        else:
            # Place all pieces that were not randomly selected in the same spot out of the scene being rendered
            piece.location = not_used_location
    
    # Choose a random lighting setup
    # Setup 1: Just 1 spot light, act as a flash from above liek real dataset
    choose_light = random.randint(0, 2)
    if choose_light == 0:
        light1.hide_render = False
        light2.hide_render = True
        light3.hide_render = True
        light4.hide_render = True
        
    if choose_light == 1:
        light1.hide_render = True
        light2.hide_render = False
        light3.hide_render = False
        light4.hide_render = True
    
    if choose_light == 2:
        light1.hide_render = True
        light2.hide_render = True
        light3.hide_render = False
        light4.hide_render = False
    
    # Render the image and store it with labels                          
    
    # Save the label file as .csv
    name = "/Users/bejay/Documents/GitHub/RecogniChess/Data Generation/Data Generated/Labels/EX_%04d" % i + ".npy"
    np.save(name, label)
    
    # Set the render settings
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = "/Users/bejay/Documents/GitHub/RecogniChess/Data Generation/Data Generated/Images/EX_%04d" % i
    bpy.ops.render.render(write_still = 1)