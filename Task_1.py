from PIL import Image
import numpy as np
# to iterate files in the directory
import os
# to sort dictionary
import operator
import matplotlib.pyplot as plt


# This function takes filename and using pillow and numpy library
# creates pixel matrix in type pf 32 bit float numbers.
# Returns a vector of image matrix.
def convert_image_pixel_matrix(filename):
    image = Image.open('/Users/onurkarakoc/Desktop/Car_Data/' + filename)
    image_array = np.asarray(image).astype('float32')
    return image_array.flatten()


# This function takes two vectors as an argument and
# computes cosine similarity.
# The formula says that cosine similarity is equal to
# dot product of two vectors over multiplication of norms of vectors.
def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_vector_a = np.linalg.norm(vector_a)
    norm_vector_b = np.linalg.norm(vector_b)
    return dot_product / (norm_vector_a * norm_vector_b)


# This function takes directory of image files and name of input image
# as an argument. For every file in the directory computes cosine similarity
# and keeps this data in dictionary. (Key: filename, value: cosine similarity).
# Dictionary is sorted in descending order after all and printed first three values.
def image_based_search(directory, source_image_name):
    filename_similarity_dictionary = {}
    # for visualization
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    source_image_vector = convert_image_pixel_matrix(source_image_name)
    for filename in os.listdir(directory):
        if filename != source_image_name:
            filename_similarity_dictionary[filename] = cosine_similarity(source_image_vector,
                                                                         convert_image_pixel_matrix(filename))
    sorted_filename_similarity_dictionary = sorted(filename_similarity_dictionary.items(), key=operator.itemgetter(1),
                                                   reverse=True)
    print('Most similar three images with similarity values for input: ', source_image_name)
    list_for_print = list(sorted_filename_similarity_dictionary)
    for i in range(3):
        print(list_for_print[i][0], ': ', list_for_print[i][1])
    # Just for visualization, separated from the code this part.
    ax1.imshow(Image.open(os.path.join('/Users/onurkarakoc/Desktop/Car_Data/', source_image_name)))
    ax1.set_title("Input")
    ax2.imshow(Image.open(os.path.join('/Users/onurkarakoc/Desktop/Car_Data/', list_for_print[0][0])))
    ax2.set_title(list_for_print[0][1])
    ax3.imshow(Image.open(os.path.join('/Users/onurkarakoc/Desktop/Car_Data/', list_for_print[1][0])))
    ax3.set_title(list_for_print[1][1])
    ax4.imshow(Image.open(os.path.join('/Users/onurkarakoc/Desktop/Car_Data/', list_for_print[2][0])))
    ax4.set_title(list_for_print[2][1])
    fig.savefig("task_" + source_image_name, dpi=200, bbox_inches='tight')


image_based_search('/Users/onurkarakoc/Desktop/Car_Data', '3952.png')
image_based_search('/Users/onurkarakoc/Desktop/Car_Data', '4228.png')
image_based_search('/Users/onurkarakoc/Desktop/Car_Data', '3861.png')
