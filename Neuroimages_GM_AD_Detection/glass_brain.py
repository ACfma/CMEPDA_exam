# -*- coding: utf-8 -*-
"""
This module will show a simple 3D rappresentation of your data.
Attention: the function will rotate the image with the permutation:
    ijk->jki
    The function WILL ALSO DISPLAY IT USING BROWSER.
"""
import numpy as np
import plotly.graph_objs as go
def glass_brain(data, opacity, surface_count, voxels, most_important = False):
    '''
    glass_brain allows you to see the 3D array as a rendered volume.
    Given the actual dataset, the matrix's indeces are permutated for\
         an optimal rappresentation.
    The image will be open with the user browser.
    Parameters
    ----------
    data : ndarray
        3D array of data to rapresent.
    opacity : float
        Sets the opacity of the surface. Opacity level over 0.25 could perform\
             as well as expected (see Plotly documentation).
    surface_count : int
        Number of isosufaces to show.High number of surfaces could leed to a\
         saturation of memory.

    Returns
    -------
    None.

    '''
    x_shape = np.linspace(0, data.shape[0]-1, data.shape[0])
    y_shape = np.linspace(0, data.shape[1]-1, data.shape[1])
    z_shape = np.linspace(0, data.shape[2]-1, data.shape[2])
    #creating grid matrix
    x_grid, y_grid, z_grid = np.meshgrid(x_shape, y_shape, z_shape)
    data_ein=np.einsum('ijk->jki', data)
    fig = go.Figure(data=go.Volume(
    x = x_grid.flatten(),
    y = y_grid.flatten(),
    z = z_grid.flatten(),
    value=data_ein.flatten(),
    isomin=data.min(),#min value of isosurface
    isomax=data.max(),#max value of isosurface
    opacity=opacity, # needs to be small to see through all surfaces
    surface_count=surface_count,
    caps=dict(x_show=False, y_show=False, z_show=False)
    ))
    fig.show(renderer="browser")
    if most_important == True:
        fig.add_scatter3d(np.einsum('ijk->jki', voxels),mode='markers',
    marker=dict(
        size=12,))
if __name__ == "__main__":
    a = np.random.rand(10,20,10)
    glass_brain(a, 0.1, 4)
