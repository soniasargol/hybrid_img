import numpy as np
import matplotlib.image as mpimg #added
import matplotlib.pyplot as plt

def my_imfilter(image, filter1):
    """
    Apply a filter to an image. Return the filtered image.

    Args
    - image: numpy nd-array of dim (m, n, c)
    - filter: numpy nd-array of dim (k, k) /--> (k, l)
    Returns
    - filtered_image: numpy nd-array of dim (m, n, c)

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to work
    with matrices is fine and encouraged. Using opencv or similar to do the
    filtering for you is not allowed.
    - I encourage you to try implementing this naively first, just be aware that
    it may take an absurdly long time to run. You will need to get a function
    that takes a reasonable amount of time to run so that the TAs can verify
    your code works.
    - Remember these are RGB images, accounting for the final image dimension.
    """


    assert filter1.shape[0] % 2 == 1
    assert filter1.shape[1] % 2 == 1

    ############################
    ### TODO: YOUR CODE HERE ###
    f_type = input("Filter type:\nzeros: 'z' - reflect: 'r'\n")
    k= filter1.shape[0]
    l = filter1.shape[1]
    print("k, l:", k, "-" , l)



    # from IPython import embed; embed()
    #print(im)
    if f_type=="z":
        print("filter:zeros")
        padded_image=np.pad(image, (((k-1)//2,(k-1)//2),((l-1)//2,(l-1)//2),(0,0)), 'constant')
        imgplot = plt.imshow(padded_image)
        # plt.show()

    else:
        print("filter:reflect")
        padded_image=np.pad(image, (((k-1)//2,(k-1)//2),((l-1)//2,(l-1)//2),(0,0)), 'reflect')
        plt.imshow(padded_image)
        # plt.show()



    # imgplot = plt.imshow(padded_image)
    # plt.show()

    filtered_image = np.zeros(image.shape)

    for i in range (image.shape[1]):
        for j in range (image.shape[0]):
            filtered_image[j,i,:]=np.sum(np.multiply(np.concatenate((filter1[...,None], filter1[...,None], filter1[...,None]), axis=2),
            padded_image[j:j+filter1.shape[0],i:i+filter1.shape[1],:]), axis=(0, 1))
            padded_image[j+filter1.shape[0]//2,i+filter1.shape[1]//2,:] = filtered_image[j,i,:]
    # filtered_image = filtered_image[filter1.shape[0]//2:-filter1.shape[0]//2, filter1.shape[1]//2:-filter1.shape[1]//2, :]



    print("padded_image.shape:",padded_image.shape)
    print("filtered_image.shape:",filtered_image.shape)

    # final_out_image = image - filtered_image +.5
    # imgplot = plt.imshow(filtered_image)
    # plt.show()
    # imgplot = plt.imshow(image)
    # plt.show()
    # imgplot = plt.imshow(padded_image)
    # plt.show()

    #for i in range (0,image.shape[0]-((filter1.shape[0]-1)*2)):


    #raise NotImplementedError('`my_imfilter` function in `student_code.py` ' +
    #'needs to be implemented')

    ### END OF STUDENT CODE ####
    ############################

    return filtered_image/np.amax(filtered_image)

def create_hybrid_image(image1, image2, filter1, filter2):
    """
    Takes two images and creates a hybrid image. Returns the low
    frequency content of image1, the high frequency content of
    image 2, and the hybrid image.

    Args
    - image1: numpy nd-array of dim (m, n, c)
    - image2: numpy nd-array of dim (m, n, c)
    Returns
    - low_frequencies: numpy nd-array of dim (m, n, c)
    - high_frequencies: numpy nd-array of dim (m, n, c)
    - hybrid_image: numpy nd-array of dim (m, n, c)

    HINTS:
    - You will use your my_imfilter function in this function.
    - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
    - If you want to use images with different dimensions, you should resize them
    in the notebook code.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    ############################
    ### TODO: YOUR CODE HERE ###


    low_frequencies=my_imfilter(image1, filter1)
    high_frequencies=(image2-my_imfilter(image2, filter2)+.5)/1.5

    # imgplot = plt.imshow(low_frequencies)
    # plt.show()
    # imgplot = plt.imshow(high_frequencies)
    # plt.show()
    hybrid_image = (high_frequencies+low_frequencies)/2
    # imgplot = plt.imshow(hybrid_image)
    # plt.show()
    # raise NotImplementedError('`create_hybrid_image` function in ' +
    # '`student_code.py` needs to be implemented')

    ### END OF STUDENT CODE ####
    ############################

    return low_frequencies, high_frequencies, hybrid_image
