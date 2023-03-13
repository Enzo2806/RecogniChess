import random
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
    if location.endswith('0') or location.endswith('8'):
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


# Variable to store the number of examples to generate
# Since we use this dataset for domain adaptation, all the examples will be in a training dataset
# All the examples will have to be labelled.
dataset_size = 1

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

# Method to select pawns
def selectPawns():
    # Randomly select white and black pawns (random number of each)
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
    # Check if piece is pawn
    if piece in all_pawns:
        # get the new location, change it if it is already occupied by a piece
        while True: 
            # Select random locaiton from available locations we computed above
            square_name, new_loc = random.choice(list(pawn_locations.items()))
            if square_name not in label:
                break
            
        piece.location = new_loc # Assign new location
        
        # Store the placed pawn and its location in label array
        newRow = [square_name, getName(piece)] # create the row to add --> "Square number", "Chess piece name"
        label = np.vstack([label, newRow]) # add the placed pawn with its square location in the label 
        
        return label
        
    # Check if piece is a bishop starting on a white square
    if piece == white_bishop_1 or piece == black_bishop_1:
        # If it is, this piece must remain on a white square so we select a random
        # white square on the board and make it it's new location
        
        # get the new location, change it if it is already occupied by a piece
        while True:
            square_name, new_loc = random.choice(list(all_white_squares.items()))
            if square_name not in label:
                break
            
        piece.location = new_loc # Assign new location
        
        # Store the placed bishop and its location in label array
        newRow = [square_name, getName(piece)] # create the row to add --> "Square number", "Chess piece name"
        label = np.vstack([label, newRow]) # add the placed bishop with its square location in the label 
        
        return label
    
    # Check if piece is a bishop starting on a green square
    if piece == white_bishop_2 or piece == black_bishop_2:
        # If it is, this pawn must remain on a green square so we select a random
        # green square on the board and make it it's new location
        
        # get the new location, change it if it is already occupied by a piece
        while True:
            square_name, new_loc = random.choice(list(all_green_squares.items()))
            if square_name not in label:
                break
            
        piece.location = new_loc # Assign new location
        
        # Store the placed bishop and its location in label array
        newRow = [square_name, getName(piece)] # create the row to add --> "Square number", "Chess piece name"
        label = np.vstack([label, newRow]) # add the placed bishop with its square location in the label 
        
        return label
        
    # Otherwise the piece can go anywhere on the board, so we just select a random location and assign it.
    else:
        
        while True:
            square_name, new_loc = random.choice(list(locations.items()))
            
        piece.location = new_loc # Assign new location
        
         # Store the placed piece and its location in label array
        newRow = [square_name, getName(piece)] # create the row to add --> "Square number", "Chess piece name"
        label = np.vstack([label, newRow]) # add the placed piece with its square location in the label 
        
        return label
        
# Array to store labels
label = ["Square number", "Chess Piece"]

# Location to put pawns we didn't randomly select
not_used_location = (6, 6, 2)

for i in range (dataset_size):
    selected_pawns =  selectPawns()
    for pawn in all_pieces:
        if pawn in selected_pawns:
            label = assignLocation(pawn, label)
        else:
            pawn.location = not_used_location

print(label.shape)
print(label)
            
    


