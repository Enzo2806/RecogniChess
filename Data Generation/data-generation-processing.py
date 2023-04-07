import numpy as np
from PIL import Image, ImageFilter
import os
import torch


# Get the dimensions of the individual squares
# These dimensions are used to crop the image into individual squares
# The dimensions are calculated by dividing the image into 8x8 squares using the Grid tool on Photoshop
# Later we make all images the same size after cropping. This is done to make the data uniform.
horizontal_top_broder = 15.5
horizontal_second_border = 154.5
horizontal_third_border = 282.4
horizontal_fourth_border = 411.4
horizontal_fifth_border = 541
horizontal_sixth_border = 670
horizontal_seventh_border = 798
horizontal_eighth_border = 926.5
horizontal_bottom_border = 1058.3

vertical_left_border = 443
vertical_second_border = 574.6
vertical_third_border = 702.6
vertical_fourth_border = 832
vertical_fifth_border = 961
vertical_sixth_border = 1089.3
vertical_seventh_border = 1219
vertical_eighth_border = 1345.1
vertical_right_border = 1478.5

# Assign the boundaries of each square to a variable
# The boundaries are used to crop the image into individual squares
# Here is how each square is defined: square = (left, upper, right, lower)
# We used the boudnaries defined above to define each square
A1_crop = (vertical_left_border, horizontal_eighth_border, vertical_second_border, horizontal_bottom_border)
A2_crop = (vertical_left_border, horizontal_seventh_border, vertical_second_border, horizontal_eighth_border)
A3_crop = (vertical_left_border, horizontal_sixth_border, vertical_second_border, horizontal_seventh_border)
A4_crop = (vertical_left_border, horizontal_fifth_border, vertical_second_border, horizontal_sixth_border)
A5_crop = (vertical_left_border, horizontal_fourth_border, vertical_second_border, horizontal_fifth_border)
A6_crop = (vertical_left_border, horizontal_third_border, vertical_second_border, horizontal_fourth_border)
A7_crop = (vertical_left_border, horizontal_second_border, vertical_second_border, horizontal_third_border)
A8_crop = (vertical_left_border, horizontal_top_broder, vertical_second_border, horizontal_second_border)
B1_crop = (vertical_second_border, horizontal_eighth_border, vertical_third_border, horizontal_bottom_border)
B2_crop = (vertical_second_border, horizontal_seventh_border, vertical_third_border, horizontal_eighth_border)
B3_crop = (vertical_second_border, horizontal_sixth_border, vertical_third_border, horizontal_seventh_border)
B4_crop = (vertical_second_border, horizontal_fifth_border, vertical_third_border, horizontal_sixth_border)
B5_crop = (vertical_second_border, horizontal_fourth_border, vertical_third_border, horizontal_fifth_border)
B6_crop = (vertical_second_border, horizontal_third_border, vertical_third_border, horizontal_fourth_border)
B7_crop = (vertical_second_border, horizontal_second_border, vertical_third_border, horizontal_third_border)
B8_crop = (vertical_second_border, horizontal_top_broder, vertical_third_border, horizontal_second_border)
C1_crop = (vertical_third_border, horizontal_eighth_border, vertical_fourth_border, horizontal_bottom_border)
C2_crop = (vertical_third_border, horizontal_seventh_border, vertical_fourth_border, horizontal_eighth_border)
C3_crop = (vertical_third_border, horizontal_sixth_border, vertical_fourth_border, horizontal_seventh_border)
C4_crop = (vertical_third_border, horizontal_fifth_border, vertical_fourth_border, horizontal_sixth_border)
C5_crop = (vertical_third_border, horizontal_fourth_border, vertical_fourth_border, horizontal_fifth_border)
C6_crop = (vertical_third_border, horizontal_third_border, vertical_fourth_border, horizontal_fourth_border)
C7_crop = (vertical_third_border, horizontal_second_border, vertical_fourth_border, horizontal_third_border)
C8_crop = (vertical_third_border, horizontal_top_broder, vertical_fourth_border, horizontal_second_border)
D1_crop = (vertical_fourth_border, horizontal_eighth_border, vertical_fifth_border, horizontal_bottom_border)
D2_crop = (vertical_fourth_border, horizontal_seventh_border, vertical_fifth_border, horizontal_eighth_border)
D3_crop = (vertical_fourth_border, horizontal_sixth_border, vertical_fifth_border, horizontal_seventh_border)
D4_crop = (vertical_fourth_border, horizontal_fifth_border, vertical_fifth_border, horizontal_sixth_border)
D5_crop = (vertical_fourth_border, horizontal_fourth_border, vertical_fifth_border, horizontal_fifth_border)
D6_crop = (vertical_fourth_border, horizontal_third_border, vertical_fifth_border, horizontal_fourth_border)
D7_crop = (vertical_fourth_border, horizontal_second_border, vertical_fifth_border, horizontal_third_border)
D8_crop = (vertical_fourth_border, horizontal_top_broder, vertical_fifth_border, horizontal_second_border)
E1_crop = (vertical_fifth_border, horizontal_eighth_border, vertical_sixth_border, horizontal_bottom_border)
E2_crop = (vertical_fifth_border, horizontal_seventh_border, vertical_sixth_border, horizontal_eighth_border)
E3_crop = (vertical_fifth_border, horizontal_sixth_border, vertical_sixth_border, horizontal_seventh_border)
E4_crop = (vertical_fifth_border, horizontal_fifth_border, vertical_sixth_border, horizontal_sixth_border)
E5_crop = (vertical_fifth_border, horizontal_fourth_border, vertical_sixth_border, horizontal_fifth_border)
E6_crop = (vertical_fifth_border, horizontal_third_border, vertical_sixth_border, horizontal_fourth_border)
E7_crop = (vertical_fifth_border, horizontal_second_border, vertical_sixth_border, horizontal_third_border)
E8_crop = (vertical_fifth_border, horizontal_top_broder, vertical_sixth_border, horizontal_second_border)
F1_crop = (vertical_sixth_border, horizontal_eighth_border, vertical_seventh_border, horizontal_bottom_border)
F2_crop = (vertical_sixth_border, horizontal_seventh_border, vertical_seventh_border, horizontal_eighth_border)
F3_crop = (vertical_sixth_border, horizontal_sixth_border, vertical_seventh_border, horizontal_seventh_border)
F4_crop = (vertical_sixth_border, horizontal_fifth_border, vertical_seventh_border, horizontal_sixth_border)
F5_crop = (vertical_sixth_border, horizontal_fourth_border, vertical_seventh_border, horizontal_fifth_border)
F6_crop = (vertical_sixth_border, horizontal_third_border, vertical_seventh_border, horizontal_fourth_border)
F7_crop = (vertical_sixth_border, horizontal_second_border, vertical_seventh_border, horizontal_third_border)
F8_crop = (vertical_sixth_border, horizontal_top_broder, vertical_seventh_border, horizontal_second_border)
G1_crop = (vertical_seventh_border, horizontal_eighth_border, vertical_eighth_border, horizontal_bottom_border)
G2_crop = (vertical_seventh_border, horizontal_seventh_border, vertical_eighth_border, horizontal_eighth_border)
G3_crop = (vertical_seventh_border, horizontal_sixth_border, vertical_eighth_border, horizontal_seventh_border)
G4_crop = (vertical_seventh_border, horizontal_fifth_border, vertical_eighth_border, horizontal_sixth_border)
G5_crop = (vertical_seventh_border, horizontal_fourth_border, vertical_eighth_border, horizontal_fifth_border)
G6_crop = (vertical_seventh_border, horizontal_third_border, vertical_eighth_border, horizontal_fourth_border)
G7_crop = (vertical_seventh_border, horizontal_second_border, vertical_eighth_border, horizontal_third_border)
G8_crop = (vertical_seventh_border, horizontal_top_broder, vertical_eighth_border, horizontal_second_border)
H1_crop = (vertical_eighth_border, horizontal_eighth_border, vertical_right_border, horizontal_bottom_border)
H2_crop = (vertical_eighth_border, horizontal_seventh_border, vertical_right_border, horizontal_eighth_border)
H3_crop = (vertical_eighth_border, horizontal_sixth_border, vertical_right_border, horizontal_seventh_border)
H4_crop = (vertical_eighth_border, horizontal_fifth_border, vertical_right_border, horizontal_sixth_border)
H5_crop = (vertical_eighth_border, horizontal_fourth_border, vertical_right_border, horizontal_fifth_border)
H6_crop = (vertical_eighth_border, horizontal_third_border, vertical_right_border, horizontal_fourth_border)
H7_crop = (vertical_eighth_border, horizontal_second_border, vertical_right_border, horizontal_third_border)
H8_crop = (vertical_eighth_border, horizontal_top_broder, vertical_right_border, horizontal_second_border)

min_range = 0
max_range = 4501
imported_range = max_range - min_range

# Initialize empty tensors for y array 
y_piece_generated = torch.Tensor(())

# Import the data generated in Data Generation/Data Generated/
label_folder = os.path.join(os.getcwd(), "Data Generation/Data Generated/Labels")
image_folder = os.path.join(os.getcwd(), "Data Generation/Data Generated/Images")

save_folder_path = os.path.join(os.getcwd(), "Data Generation/Pre Processed Data Generated/Square Images/")

def convert_label(label_square):
    # Convert the label of the square to a fixed number following this mapping:
    # Piece to square label conversion:
    # Empty: 0, 
    # White pawn: 1,
    # White knight: 2, 
    # White bishop: 3, 
    # White rook: 4, 
    # White queen: 5, 
    # White king: 6, 
    # Black pawn: 7, 
    # Black knight: 8, 
    # Black bishop: 9, 
    # Black rook: 10, 
    # Black queen: 11, 
    # Black king: 12
    if label_square == '':
        label_square = 0
    elif label_square == 'White Pawn':
        label_square = 1
    elif label_square == 'White Knight':
        label_square = 2
    elif label_square == 'White Bishop':
        label_square = 3
    elif label_square == 'White Rook':
        label_square = 4
    elif label_square == 'White Queen':
        label_square = 5
    elif label_square == 'White King':
        label_square = 6
    elif label_square == 'Black Pawn':
        label_square = 7
    elif label_square == 'Black Knight':
        label_square = 8
    elif label_square == 'Black Bishop':
        label_square = 9
    elif label_square == 'Black Rook':
        label_square = 10
    elif label_square == 'Black Queen':
        label_square = 11
    elif label_square == 'Black King':
        label_square = 12
    return label_square

np_convert_label = np.vectorize(convert_label)

# Loop over training examples
for i in range (min_range, max_range):
    # Set the path to the label and image
    label_path = os.path.join(label_folder, "EX_%04d.npy" % i)
    image_path = os.path.join(image_folder, "EX_%04d.png" % i)

    # Load the image and label
    label = np.load(label_path)
    image = Image.open(image_path)
    # Convert to RGB color space, remove alpha channel
    image = image.convert('RGB')
    # Blur the image to mimic real life dataset
    image = image.filter(ImageFilter.GaussianBlur(radius=1))

    # Crop the image into 64 pieces
    # Store each cropped square into a variable
    A1 = image.crop(A1_crop)
    A2 = image.crop(A2_crop)
    A3 = image.crop(A3_crop)
    A4 = image.crop(A4_crop)
    A5 = image.crop(A5_crop)
    A6 = image.crop(A6_crop)
    A7 = image.crop(A7_crop)
    A8 = image.crop(A8_crop)
    B1 = image.crop(B1_crop)
    B2 = image.crop(B2_crop)
    B3 = image.crop(B3_crop)
    B4 = image.crop(B4_crop)
    B5 = image.crop(B5_crop)
    B6 = image.crop(B6_crop)
    B7 = image.crop(B7_crop)
    B8 = image.crop(B8_crop)
    C1 = image.crop(C1_crop)
    C2 = image.crop(C2_crop)
    C3 = image.crop(C3_crop)
    C4 = image.crop(C4_crop)
    C5 = image.crop(C5_crop)
    C6 = image.crop(C6_crop)
    C7 = image.crop(C7_crop)
    C8 = image.crop(C8_crop)
    D1 = image.crop(D1_crop)
    D2 = image.crop(D2_crop)
    D3 = image.crop(D3_crop)
    D4 = image.crop(D4_crop)
    D5 = image.crop(D5_crop)
    D6 = image.crop(D6_crop)
    D7 = image.crop(D7_crop)
    D8 = image.crop(D8_crop)
    E1 = image.crop(E1_crop)
    E2 = image.crop(E2_crop)
    E3 = image.crop(E3_crop)
    E4 = image.crop(E4_crop)
    E5 = image.crop(E5_crop)
    E6 = image.crop(E6_crop)
    E7 = image.crop(E7_crop)
    E8 = image.crop(E8_crop)
    F1 = image.crop(F1_crop)
    F2 = image.crop(F2_crop)
    F3 = image.crop(F3_crop)
    F4 = image.crop(F4_crop)
    F5 = image.crop(F5_crop)
    F6 = image.crop(F6_crop)
    F7 = image.crop(F7_crop)
    F8 = image.crop(F8_crop)
    G1 = image.crop(G1_crop)
    G2 = image.crop(G2_crop)
    G3 = image.crop(G3_crop)
    G4 = image.crop(G4_crop)
    G5 = image.crop(G5_crop)
    G6 = image.crop(G6_crop)
    G7 = image.crop(G7_crop)
    G8 = image.crop(G8_crop)
    H1 = image.crop(H1_crop)
    H2 = image.crop(H2_crop)
    H3 = image.crop(H3_crop)
    H4 = image.crop(H4_crop)
    H5 = image.crop(H5_crop)
    H6 = image.crop(H6_crop)
    H7 = image.crop(H7_crop)
    H8 = image.crop(H8_crop)

    # create an array with all the cropped images:
    images = [A1, A2, A3, A4, A5, A6, A7, A8, B1, B2, B3, B4, B5, B6, B7, B8, C1, C2, C3, C4, C5, C6, C7, C8, D1, D2, D3, D4, D5, D6, D7, D8, E1, E2, E3, E4, E5, E6, E7, E8, F1, F2, F3, F4, F5, F6, F7, F8, G1, G2, G3, G4, G5, G6, G7, G8, H1, H2, H3, H4, H5, H6, H7, H8]
    
    y_piece_generated = torch.cat((y_piece_generated, torch.tensor(np_convert_label(label[1:, 1]), dtype=torch.long)))

    # loop through all the images and resize them
    # to 100x100 pixels
    # Save them in new folder in array
    for j, img in enumerate(images):
        # resize all images to 100x100 pixels
        img = img.resize((100, 100))
        
        # Save the image in the new folder
        img.save(save_folder_path+'EX_%06d' % (i*64+j) + '.png')

# Save the label in the new folder
torch.save(y_piece_generated, save_folder_path+'y_generated.pt')
