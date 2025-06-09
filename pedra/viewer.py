# %matplotlib qt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

from cana.util import kwargupdate


class CardinalViewer:
    r"""
    Class to viasualize sky cardinal points on image.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The figure object for the plot.
    ax : matplotlib.axes.Axes
        The axes object for the plot.
    directions: str
        Cardinal directions that will be ploted. 
    """

    def __init__(self, directions='NSEW'):
        r"""
        Initialize CardinalViewer class.

        Parameters
        ----------
        directions: str
            Cardinal directions that will be ploted. 
        """
        # directions
        self.directions = [*directions]
        self.directions_positions = {'N': (0, 1),
                                     'S': (0, -1),
                                     'E': (1, 0),
                                     'W': (-1, 0)}
        for d in directions:
            assert d in self.directions_positions.keys(), f"{d} not accepted. N S E W only."
        
        # Define location map with normalized positions
        self.loc_map = {'center': (0.5, 0.5),
                        'upper left': (0.15, 0.85),
                        'upper right': (0.85, 0.85),
                        'lower left': (0.15, 0.15),
                        'lower right': (0.85, 0.15),
                        'upper center': (0.5, 0.85),
                        'lower center': (0.5, 0.15),
                        'center left': (0.15, 0.5),
                        'center right': (0.85, 0.5)}

    
    def plot(self, ax=None, fig=None, 
             loc='lower right', angle=0, size=0.05, 
             label_spacer=0.075, label_kwargs=None, **kwargs):
        r"""
        Plot the cardinal directions.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object for the plot.

        fig : matplotlib.figure.Figure
            The figure object for the plot.

        loc: str or tuple
            The location of the cardinal directions on the image.
            Accepted values are 'upper left', 'upper right', 'lower left', 'lower right', 
            'upper center', 'lower center', 'center left', 'center right' and 'center'.
            The location can also be a 2-tuple giving the coordinates for the plot.

        angle: float
            The rotation angle for the cardinal plot. 

        size: float
            Arrows size.

        label_spacer: float
            The space between the arrows and labels of the cardinal points. 

        label_kwargs: dict (optional)
            Matplotlib .text parameters for the cardinal points labels. 

        **kwargs
            Matplotlib .arrow parameters for the cardinal points arrows.
        
        Returns
        -------
        fig, ax
        """
        # matplotlib figure and axis
        if fig is None:
            fig = plt.gcf()
        if ax is None:
            ax = plt.gca()
        # Get the axis limits
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        # setting default values for arrows plot with matplotlib
        kwargs_defaults = {'head_width': 0.02 * (x_max - x_min), 
                           'head_length': 0.03 * (y_max - y_min), 
                           'fc': '#FF674D',
                           'alpha': 0.9, 
                           'ec': '#FF674D'}
        kwargs = kwargupdate(kwargs_defaults, kwargs)

        label_kwargs_defaults = {'fontsize': 12,
                                 'color': '#FF674D',
                                 'ha': 'center'}
        label_kwargs = kwargupdate(label_kwargs_defaults, label_kwargs)
        # Location of cardinal plot
        if isinstance(loc, str):
            if loc in self.loc_map:
                x, y = self.loc_map[loc]
                origin = (x_min + x * (x_max - x_min), y_min + y * (y_max - y_min))
        else:
            origin = loc
        # Adjust arrow length based on axis limits
        arrow_length_x = size * (x_max - x_min)
        arrow_length_y = size * (y_max - y_min)
        # Rotation matrix
        theta = np.radians(angle)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        # Plot each arrow with rotation
        for direction in self.directions:
            rotated_vector = np.dot(rotation_matrix, self.directions_positions[direction])
            ax.arrow(origin[0], origin[1], 
                     rotated_vector[0] * arrow_length_x, 
                     rotated_vector[1] * arrow_length_y,
                     **kwargs)
            ax.text(origin[0] + rotated_vector[0] * (arrow_length_x + label_spacer * (x_max - x_min)), 
                    origin[1] + rotated_vector[1] * (arrow_length_y + label_spacer * (y_max - y_min)), 
                    direction, **label_kwargs)
        return fig, ax


class ImageViewer:
    """
    A class to display an image with adjustable contrast sliders.

    Attributes
    ----------
    image : pedra.Image
        The image to be displayed.
    cmap : str
        The colormap to be used for displaying the image.
    vmin : float
        The minimum value for the colormap.
    vmax : float
        The maximum value for the colormap.
    backend : str
        The backend used by matplotlib.
    fig : matplotlib.figure.Figure
        The figure object for the plot.
    ax : matplotlib.axes.Axes
        The axes object for the plot.
    im : matplotlib.image.AxesImage
        The image object for the plot.
    """

    def __init__(self, wcs=True, 
                 cardinal=None,
                 sundirection=True):
        """
        Initializes the ImageShower with the given image and colormap.

        Parameters:
        image : np.ndarray
            The image to be displayed.
        cmap : str, optional
            The colormap to be used for displaying the image (default is 'gray').
        """
        self.wcs = wcs
        self.cardinal = cardinal
        self.sundirection = sundirection
        # Determine the backend
        self.backend = plt.get_backend()
        # placeholders
        self.vmin = None
        self.vmax = None
        self.ax = None
        self.fig = None
        self.im = None

    def __call__(self, image, ax=None, fig=None,
             cardinal_kwargs=None, 
             label_kwargs=None,
             **kwargs):
        r"""
        Plot the cardinal directions.

        ax : matplotlib.axes.Axes
            The axes object for the plot.

        fig : matplotlib.figure.Figure
            The figure object for the plot.
        """ 
        label_kwargs_defaults = {'fontsize': 14}
        label_kwargs = kwargupdate(label_kwargs_defaults, label_kwargs)
        # matplotlib figure and axis
        if fig is None:
            self.fig = plt.gcf()
        else:
            self.fig = fig
        if ax is None:
            if (image.wcs is not None) & self.wcs:
                self.ax = plt.subplot(projection=image.wcs)
            else:
                self.ax = plt.gca()
                self.ax.set_xlabel('X (px)', **label_kwargs)
                self.ax.set_ylabel('Y (px)', **label_kwargs)
        else:
            self.ax = ax
        self.ax.set_title(image.label)
        # setting default values for image plot with matplotlib
        self.vmin, self.vmax = np.nanpercentile(image.data, (2, 98))
        kwargs_defaults = {'cmap': plt.cm.gray, 
                           'vmin': self.vmin,
                           'vmax': self.vmax,
                           'origin': 'lower'}
        kwargs = kwargupdate(kwargs_defaults, kwargs)
        # Plot image
        self.im = self.ax.imshow(image.data, **kwargs)
        # Cardinal points
        if self.cardinal is not None:
            if cardinal_kwargs is None:
                cardinal_kwargs = {}   ## -> implement this
            caview = CardinalViewer(self.cardinal)
            caview.plot(fig=fig, ax=self.ax,
                        angle = image.north_angle(),
                        **cardinal_kwargs)
        return self.fig, self.ax


class SourceViewer:
    r"""
    A class to display an image with adjustable contrast sliders.

    Attributes
    ----------
    image : pedra.Image
        The image to be displayed.
    cmap : str
        The colormap to be used for displaying the image.
    vmin : float
        The minimum value for the colormap.
    vmax : float
        The maximum value for the colormap.
    backend : str
        The backend used by matplotlib.
    fig : matplotlib.figure.Figure
        The figure object for the plot.
    ax : matplotlib.axes.Axes
        The axes object for the plot.
    im : matplotlib.image.AxesImage
        The image object for the plot.
    """

    def __init__(self):
        """
        Initializes the ImageShower with the given image and colormap.

        Parameters:
        image : np.ndarray
            The image to be displayed.
        cmap : str, optional
            The colormap to be used for displaying the image (default is 'gray').
        """
        # Determine the backend
        self.backend = plt.get_backend()
        # placeholders
        self.ax = None
        self.fig = None
        self.im = None
        

    def __call__(self, source, ax=None, fig=None,
             label_kwargs=None,
             **kwargs):
        r"""
        Plot the cardinal directions.

        ax : matplotlib.axes.Axes
            The axes object for the plot.

        fig : matplotlib.figure.Figure
            The figure object for the plot.
        """ 
        label_kwargs_defaults = {'fontsize': 14}
        label_kwargs = kwargupdate(label_kwargs_defaults, label_kwargs)
        # matplotlib figure and axis
        if fig is None:
            self.fig = plt.gcf()
        else:
            self.fig = fig
        if ax is None:
            self.ax = plt.subplot(projection='3d')
        else:
            self.ax = ax
        # self.ax.set_title(image.label)
        # setting default values for image plot with matplotlib
        # self.vmin, self.vmax = np.nanpercentile(image.data, (2, 98))
        kwargs_defaults = {'cmap': plt.cm.gray, 
                           'origin': 'lower'}
        kwargs = kwargupdate(kwargs_defaults, kwargs)
        # Plot source
        x = np.arange(source.img.data.shape[1])
        y = np.arange(source.img.data.shape[0])
        x, y = np.meshgrid(x, y)
        self.ax.plot_surface(x, y, source.img.data, cmap='viridis', edgecolor='none', alpha=0.5)

        return self.fig, self.ax
    

    def view_fit(self, source, 
                 label_kwargs=None, axis='xy',
                 **kwargs):
        r"""
        """
        label_kwargs_defaults = {'fontsize': 14}
        label_kwargs = kwargupdate(label_kwargs_defaults, label_kwargs)
        # matplotlib figure and axis
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # Define a 2-row grid
        ax_main = fig.add_subplot(gs[0])
        ax_res = fig.add_subplot(gs[1], sharex=ax_main)  # Share the x-axis with the main plot
        if 'x' in axis:
            x_data = source.img.sum(axis=0)
            x_fit = source.fit.sum(axis=0)
            ax_main.plot(range(x_data.shape[0]), x_data, lw=1.5, alpha=0.8, label='X')
            ax_main.plot(range(x_fit.shape[0]), x_fit,  '-.', lw=1.5, alpha=0.8, label='X fit')
            ax_res.plot(range(x_fit.shape[0]), x_data - x_fit, lw=1.5, alpha=0.8, label='X Residual')

        if 'y' in axis:
            y_data = source.img.sum(axis=1)
            y_fit = source.fit.sum(axis=1)
            ax_main.plot(range(y_data.shape[0]), y_data, lw=1.5, alpha=0.8, label='Y')
            ax_main.plot(range(y_fit.shape[0]), y_fit, '-.', lw=1.5, alpha=0.8, label='Y fit')
            ax_res.plot(range(y_fit.shape[0]), y_data - y_fit, '-.', lw=1.5, alpha=0.8, label='Y Residual')
       
        ax_res.set_xlabel("Axis (px)", **label_kwargs)
        ax_main.set_ylabel("Counts", **label_kwargs)

        ax_main.legend()
        ax_res.legend(ncols=2)
        return fig, ax_main, ax_res

    def view_fit3D(self, source, ax=None, fig=None,
             label_kwargs=None,
             **kwargs):
        r"""
        Plot the cardinal directions.

        ax : matplotlib.axes.Axes
            The axes object for the plot.

        fig : matplotlib.figure.Figure
            The figure object for the plot.
        """ 
        label_kwargs_defaults = {'fontsize': 14}
        label_kwargs = kwargupdate(label_kwargs_defaults, label_kwargs)
        # matplotlib figure and axis
        if fig is None:
            self.fig = plt.gcf()
        else:
            self.fig = fig
        if ax is None:
            self.ax = plt.subplot(projection='3d')
        else:
            self.ax = ax
        # self.ax.set_title(image.label)
        # setting default values for image plot with matplotlib
        # self.vmin, self.vmax = np.nanpercentile(image.data, (2, 98))
        kwargs_defaults = {'color': 'firebrick', 
                           'alpha': 0.5,
                           'rstride': 3,
                           'cstride': 3}
        kwargs = kwargupdate(kwargs_defaults, kwargs)
        # Plot source
        x = np.arange(source.img.data.shape[1])
        y = np.arange(source.img.data.shape[0])
        x, y = np.meshgrid(x, y)
        self.ax.plot_surface(x, y, source.img.data, cmap='viridis', edgecolor='none', alpha=0.5)
        self.ax.plot_wireframe(x, y, source.fit.data, **kwargs)

        return self.fig, self.ax