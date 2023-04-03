import numpy as np
from PIL import Image
import cv2


# Import the data generated in Data Generation/Data Generated/
for i in range (4501):
    # Set the path to the label and image
    labelpath= "/Users/bejay/Documents/GitHub/RecogniChess/Data Generation/Data Generated/Labels/EX_%04d" % i + ".npy"
    imagepath= "/Users/bejay/Documents/GitHub/RecogniChess/Data Generation/Data Generated/Images/EX_%04d" % i + ".png"

    # Load the image and label
    label = np.load(labelpath)
    image = Image.open(imagepath)

    image = np.asarray(image)

    image = cv2.blur(image, (5,5))

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
    A1 = (vertical_left_border, horizontal_eighth_border, vertical_second_border, horizontal_bottom_border)
    A2 = (vertical_left_border, horizontal_seventh_border, vertical_second_border, horizontal_eighth_border)
    A3 = (vertical_left_border, horizontal_sixth_border, vertical_second_border, horizontal_seventh_border)
    A4 = (vertical_left_border, horizontal_fifth_border, vertical_second_border, horizontal_sixth_border)
    A5 = (vertical_left_border, horizontal_fourth_border, vertical_second_border, horizontal_fifth_border)
    A6 = (vertical_left_border, horizontal_third_border, vertical_second_border, horizontal_fourth_border)
    A7 = (vertical_left_border, horizontal_second_border, vertical_second_border, horizontal_third_border)
    A8 = (vertical_left_border, horizontal_top_broder, vertical_second_border, horizontal_second_border)
    B1 = (vertical_second_border, horizontal_eighth_border, vertical_third_border, horizontal_bottom_border)
    B2 = (vertical_second_border, horizontal_seventh_border, vertical_third_border, horizontal_eighth_border)
    B3 = (vertical_second_border, horizontal_sixth_border, vertical_third_border, horizontal_seventh_border)
    B4 = (vertical_second_border, horizontal_fifth_border, vertical_third_border, horizontal_sixth_border)
    B5 = (vertical_second_border, horizontal_fourth_border, vertical_third_border, horizontal_fifth_border)
    B6 = (vertical_second_border, horizontal_third_border, vertical_third_border, horizontal_fourth_border)
    B7 = (vertical_second_border, horizontal_second_border, vertical_third_border, horizontal_third_border)
    B8 = (vertical_second_border, horizontal_top_broder, vertical_third_border, horizontal_second_border)
    C1 = (vertical_third_border, horizontal_eighth_border, vertical_fourth_border, horizontal_bottom_border)
    C2 = (vertical_third_border, horizontal_seventh_border, vertical_fourth_border, horizontal_eighth_border)
    C3 = (vertical_third_border, horizontal_sixth_border, vertical_fourth_border, horizontal_seventh_border)
    C4 = (vertical_third_border, horizontal_fifth_border, vertical_fourth_border, horizontal_sixth_border)
    C5 = (vertical_third_border, horizontal_fourth_border, vertical_fourth_border, horizontal_fifth_border)
    C6 = (vertical_third_border, horizontal_third_border, vertical_fourth_border, horizontal_fourth_border)
    C7 = (vertical_third_border, horizontal_second_border, vertical_fourth_border, horizontal_third_border)
    C8 = (vertical_third_border, horizontal_top_broder, vertical_fourth_border, horizontal_second_border)
    D1 = (vertical_fourth_border, horizontal_eighth_border, vertical_fifth_border, horizontal_bottom_border)
    D2 = (vertical_fourth_border, horizontal_seventh_border, vertical_fifth_border, horizontal_eighth_border)
    D3 = (vertical_fourth_border, horizontal_sixth_border, vertical_fifth_border, horizontal_seventh_border)
    D4 = (vertical_fourth_border, horizontal_fifth_border, vertical_fifth_border, horizontal_sixth_border)
    D5 = (vertical_fourth_border, horizontal_fourth_border, vertical_fifth_border, horizontal_fifth_border)
    D6 = (vertical_fourth_border, horizontal_third_border, vertical_fifth_border, horizontal_fourth_border)
    D7 = (vertical_fourth_border, horizontal_second_border, vertical_fifth_border, horizontal_third_border)
    D8 = (vertical_fourth_border, horizontal_top_broder, vertical_fifth_border, horizontal_second_border)
    E1 = (vertical_fifth_border, horizontal_eighth_border, vertical_sixth_border, horizontal_bottom_border)
    E2 = (vertical_fifth_border, horizontal_seventh_border, vertical_sixth_border, horizontal_eighth_border)
    E3 = (vertical_fifth_border, horizontal_sixth_border, vertical_sixth_border, horizontal_seventh_border)
    E4 = (vertical_fifth_border, horizontal_fifth_border, vertical_sixth_border, horizontal_sixth_border)
    E5 = (vertical_fifth_border, horizontal_fourth_border, vertical_sixth_border, horizontal_fifth_border)
    E6 = (vertical_fifth_border, horizontal_third_border, vertical_sixth_border, horizontal_fourth_border)
    E7 = (vertical_fifth_border, horizontal_second_border, vertical_sixth_border, horizontal_third_border)
    E8 = (vertical_fifth_border, horizontal_top_broder, vertical_sixth_border, horizontal_second_border)
    F1 = (vertical_sixth_border, horizontal_eighth_border, vertical_seventh_border, horizontal_bottom_border)
    F2 = (vertical_sixth_border, horizontal_seventh_border, vertical_seventh_border, horizontal_eighth_border)
    F3 = (vertical_sixth_border, horizontal_sixth_border, vertical_seventh_border, horizontal_seventh_border)
    F4 = (vertical_sixth_border, horizontal_fifth_border, vertical_seventh_border, horizontal_sixth_border)
    F5 = (vertical_sixth_border, horizontal_fourth_border, vertical_seventh_border, horizontal_fifth_border)
    F6 = (vertical_sixth_border, horizontal_third_border, vertical_seventh_border, horizontal_fourth_border)
    F7 = (vertical_sixth_border, horizontal_second_border, vertical_seventh_border, horizontal_third_border)
    F8 = (vertical_sixth_border, horizontal_top_broder, vertical_seventh_border, horizontal_second_border)
    G1 = (vertical_seventh_border, horizontal_eighth_border, vertical_eighth_border, horizontal_bottom_border)
    G2 = (vertical_seventh_border, horizontal_seventh_border, vertical_eighth_border, horizontal_eighth_border)
    G3 = (vertical_seventh_border, horizontal_sixth_border, vertical_eighth_border, horizontal_seventh_border)
    G4 = (vertical_seventh_border, horizontal_fifth_border, vertical_eighth_border, horizontal_sixth_border)
    G5 = (vertical_seventh_border, horizontal_fourth_border, vertical_eighth_border, horizontal_fifth_border)
    G6 = (vertical_seventh_border, horizontal_third_border, vertical_eighth_border, horizontal_fourth_border)
    G7 = (vertical_seventh_border, horizontal_second_border, vertical_eighth_border, horizontal_third_border)
    G8 = (vertical_seventh_border, horizontal_top_broder, vertical_eighth_border, horizontal_second_border)
    H1 = (vertical_eighth_border, horizontal_eighth_border, vertical_right_border, horizontal_bottom_border)
    H2 = (vertical_eighth_border, horizontal_seventh_border, vertical_right_border, horizontal_eighth_border)
    H3 = (vertical_eighth_border, horizontal_sixth_border, vertical_right_border, horizontal_seventh_border)
    H4 = (vertical_eighth_border, horizontal_fifth_border, vertical_right_border, horizontal_sixth_border)
    H5 = (vertical_eighth_border, horizontal_fourth_border, vertical_right_border, horizontal_fifth_border)
    H6 = (vertical_eighth_border, horizontal_third_border, vertical_right_border, horizontal_fourth_border)
    H7 = (vertical_eighth_border, horizontal_second_border, vertical_right_border, horizontal_third_border)
    H8 = (vertical_eighth_border, horizontal_top_broder, vertical_right_border, horizontal_second_border)

    # Reconvert the image to pillow  from numpy
    image = Image.fromarray(image)

    # Crop the image into 64 pieces
    # Store each cropped square into a variable
    A1 = image.crop(A1)
    A2 = image.crop(A2)
    A3 = image.crop(A3)
    A4 = image.crop(A4)
    A5 = image.crop(A5)
    A6 = image.crop(A6)
    A7 = image.crop(A7)
    A8 = image.crop(A8)
    B1 = image.crop(B1)
    B2 = image.crop(B2)
    B3 = image.crop(B3)
    B4 = image.crop(B4)
    B5 = image.crop(B5)
    B6 = image.crop(B6)
    B7 = image.crop(B7)
    B8 = image.crop(B8)
    C1 = image.crop(C1)
    C2 = image.crop(C2)
    C3 = image.crop(C3)
    C4 = image.crop(C4)
    C5 = image.crop(C5)
    C6 = image.crop(C6)
    C7 = image.crop(C7)
    C8 = image.crop(C8)
    D1 = image.crop(D1)
    D2 = image.crop(D2)
    D3 = image.crop(D3)
    D4 = image.crop(D4)
    D5 = image.crop(D5)
    D6 = image.crop(D6)
    D7 = image.crop(D7)
    D8 = image.crop(D8)
    E1 = image.crop(E1)
    E2 = image.crop(E2)
    E3 = image.crop(E3)
    E4 = image.crop(E4)
    E5 = image.crop(E5)
    E6 = image.crop(E6)
    E7 = image.crop(E7)
    E8 = image.crop(E8)
    F1 = image.crop(F1)
    F2 = image.crop(F2)
    F3 = image.crop(F3)
    F4 = image.crop(F4)
    F5 = image.crop(F5)
    F6 = image.crop(F6)
    F7 = image.crop(F7)
    F8 = image.crop(F8)
    G1 = image.crop(G1)
    G2 = image.crop(G2)
    G3 = image.crop(G3)
    G4 = image.crop(G4)
    G5 = image.crop(G5)
    G6 = image.crop(G6)
    G7 = image.crop(G7)
    G8 = image.crop(G8)
    H1 = image.crop(H1)
    H2 = image.crop(H2)
    H3 = image.crop(H3)
    H4 = image.crop(H4)
    H5 = image.crop(H5)
    H6 = image.crop(H6)
    H7 = image.crop(H7)
    H8 = image.crop(H8)

    # create an array with all the cropped images:
    images = [A1, A2, A3, A4, A5, A6, A7, A8, B1, B2, B3, B4, B5, B6, B7, B8, C1, C2, C3, C4, C5, C6, C7, C8, D1, D2, D3, D4, D5, D6, D7, D8, E1, E2, E3, E4, E5, E6, E7, E8, F1, F2, F3, F4, F5, F6, F7, F8, G1, G2, G3, G4, G5, G6, G7, G8, H1, H2, H3, H4, H5, H6, H7, H8]
    
    # Create an empty array to store this training exmaples' images and their correpsonding labels / square emplacement names
    # We store a new array for each training example, eahc array has all square names, the image of the square and label
    # We will append the labels and images to this array in the following loop
    final_array = np.array(["Square Name", "Image", "Piece Label"], dtype = object)

    # compute the average size of the images
    # total_width = 0
    # total_height = 0
    # for img in images:
    #     total_width += img.size[0]
    #     total_height += img.size[1]
    # average_width = total_width / len(images)
    # average_height = total_height / len(images)
    # print(average_width, average_height)
    # RESULT is approximately 130x130, so we will resize all images to 130x130

    # loop through all the images and resize them
    # to 130x130 (computed average size above)
    # Save them in new folder
    for j, img in enumerate(images):
        # resize all images to 130x130 pixels
        # Causes warning, ignore it
        img = img.resize((130, 130), Image.ANTIALIAS)

        # Get the square name of the image to save it in the new folder
        square_name = label[j+1][0]

        # GOAL: Save a numpy array containing the training example number, image, the label of the image and the square name (one hot encoded)

        # get the label of the pawn shown in image
        label_square = label[j+1][1]

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
        

        # Create a numpy array with the tarining example name, the image, the label of the image and the square name (one hot encoded))
        # Concatenate this new numpy array to the global numpy array we will save later
        final_array = np.vstack([final_array, np.array([square_name, np.array(img), label_square], dtype = object)])

    # Save the numpy array containing all images, labels and square names in a new folder
    label_folder_path = "/Users/bejay/Documents/GitHub/RecogniChess/Data Generation/Data Generated/Dataset PreProcessed/EX_%04d"%i+".npy"
    np.save(label_folder_path, final_array, allow_pickle=True)

