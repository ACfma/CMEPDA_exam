"""
brain_animation creates an animation over one of the three dimension in order \
to better visualize your data.
As standalone program, it will visualize a random array od shape (121,145,121).
"""
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation

import numpy as np

def brain_animation(image, interval, delay):
    '''
    brain_animation will create a simple animation of the blain along the three\
     main axes of a given nifti image.

    Parameters
    ----------
    Image: array
        3D selected nifti image.

    interval: int
        Time (in ms) between frames.

    delay: int
        Time of sleep (in ms) before repeating the animation.

    Returns
    -------
    Animation: Matplotlib.animation object
        Return Matplotlib.animation object along with the plotted animation\
         when assigned (as specified in Matplotlib documentation).
    '''
    def brain_sequence(type_of_scan,data):
        '''
        Brain_Sequence returns a list of frames from a 3D ndarray.

        Parameters
        ----------
        type_of_scan: string
            Specified view of the array: "Axial", "Coronal" or "Sagittal".
        data: array
            3D array to show

        Returns
        -------
            imgs: list of AxesImage
                List of frames to be animated.
        '''
        imgs=[]
        if type_of_scan == 'Axial':
            for i, _ in enumerate(data[:,0,0]):
                img = plt.imshow(data[i,:,:], animated = True)
                imgs.append([img])
        elif type_of_scan == 'Coronal':
            for i, _ in enumerate(data[0,:,0]):
                img = plt.imshow(data[:,i,:], animated = True)
                imgs.append([img])
        elif type_of_scan == 'Sagittal':
            for i, _ in enumerate(data[0,0,:]):
                img = plt.imshow(data[:,:,i], animated = True)
                imgs.append([img])
        return imgs

    type_of_scan = input('\nType your view animation (Axial/Coronal/Sagittal): ')
    fig = plt.figure('Brain scan')
    return  ArtistAnimation(fig, brain_sequence(type_of_scan, image),
                            interval=interval, blit=True, repeat_delay=delay)

if __name__ == "__main__":
    a = np.random.rand(121,145,121)
    anim = brain_animation(a, 50, 100)
