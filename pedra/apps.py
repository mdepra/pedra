import matplotlib.pyplot as plt
import numpy as np
from .viewer import ImageViewer
from .sources import SourcesDataFrame

from matplotlib.widgets import Button, Slider, RectangleSelector
import time
from threading import Thread


class ContrastViewer(ImageViewer):
    r"""
    """

    def __init__(self, wcs=True, cardinal=None, sundirection=True):
        super().__init__(wcs, cardinal, sundirection)

        self.positions = []
        self.scatter_plots = []
        self.scatter_selector = None
        self.scattersize = 50
        self.rect_selector = None
        self.rect_verts = None
        self.rect_area = None
        self.image = None


    def __call__(self, image, 
                 ax_im=None, ax_sliders=None, fig=None,
                 cardinal_kwargs=None, label_kwargs=None,
                 **kwargs):
        r"""
        Plot the cardinal directions.

        ax : matplotlib.axes.Axes
            The axes object for the plot.

        fig : matplotlib.figure.Figure
            The figure object for the plot.
        """
        self.image = image
        if fig is None:
            fig = plt.figure()
            plt.tight_layout()
        if ax_im is None:
            gs = fig.add_gridspec(1, 3, width_ratios=[10, 0.75, 0.75])
            if (image.wcs is not None) & self.wcs:
                ax_im = fig.add_subplot(gs[0, 0], projection=image.wcs)
            else:
                ax_im = fig.add_subplot(gs[0, 0])
                ax_im.set_xlabel('X (px)')
                ax_im.set_ylabel('Y (px)')
            ax_sliders = [fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]

        super().__call__(image, ax=ax_im, fig=fig,
                         cardinal_kwargs=cardinal_kwargs, label_kwargs=label_kwargs,
                         **kwargs)
        self.constrast_sliders(image, ax_sliders)
        plt.connect('key_press_event', self.toggle_selector)
        
        return fig, ax_im, ax_sliders
    
    def select_scatter(self, radius=None, **kwargs):
        r"""
        """
        self.scatter_kwargs = None
        if radius is not None:
            self.scattersize = 50
        self.scatter_selector = self.fig.canvas.mpl_connect('button_press_event', self.on_click)


    def constrast_sliders(self, image, ax=None):
        r"""
        """
        ax_vmin, ax_vmax = ax
        slider_limit_low, slider_limit_upp = np.nanpercentile(image.data, (0.1, 99))
        # Use matplotlib sliders for external window (Qt)
        self.s_vmin = Slider(ax_vmin, 'vmin', 
                                slider_limit_low, 
                                slider_limit_upp, 
                                valinit=self.vmin, 
                                orientation='vertical')
        self.s_vmax = Slider(ax_vmax, 'vmax', 
                                slider_limit_low, 
                                slider_limit_upp, 
                                valinit=self.vmax, 
                                orientation='vertical')
        
        self.s_vmin.on_changed(self.update_matplotlib)
        self.s_vmax.on_changed(self.update_matplotlib)
        # Adjust layout
        plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1, wspace=0.2, hspace=0.2)
        return self.fig, self.ax
             

    def update_matplotlib(self, val):
        """
        Update the colormap limits when the matplotlib sliders change.

        Parameters:
        val : float
            The new value from the matplotlib Slider event.
        """
        if self.s_vmin.val < self.s_vmax.val:
            self.vmin = self.s_vmin.val
        if self.s_vmax.val > self.s_vmin.val:
            self.vmax = self.s_vmax.val
        # self.vmin = self.s_vmin.val
        self.im.set_clim(vmin=self.vmin, vmax=self.vmax)
        self.fig.canvas.draw_idle()
    
    def on_click(self, event):
        if self.rect_selector is not None:
            if self.rect_selector.active:
                return
        if event.inaxes == self.ax:
            x, y = event.xdata, event.ydata
            if event.button == 1:  # Left click
                self.positions.append((x, y))
                scatter = self.ax.scatter(x, y, c='none', s=self.scattersize, edgecolors='steelblue', lw=2)
                self.scatter_plots.append(scatter)
            elif event.button == 3:  # Right click
                self.positions = []
                for scatter in self.scatter_plots:
                    scatter.remove()
                self.scatter_plots = []
            self.fig.canvas.draw_idle()

    def select_rectangle(self):
        if self.rect_selector is not None:
            self.rect_selector.set_active(True)
        else:
            self.rect_selector = RectangleSelector(self.ax, self.on_select,
                                                   useblit=True,
                                                   button=[1],  # left mouse button
                                                   minspanx=5, minspany=5,
                                                   spancoords='pixels',
                                                   interactive=True)
        # plt.connect('key_press_event', self.toggle_selector)

    def on_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.rect_verts = np.int32([x1, x2, y1, y2])
        self.rect_area = abs(x2 - x1) * abs(y2 - y1)
        print(f"Rectangle selected with vertices: {self.rect_verts} and area: {self.rect_area}")

    def toggle_selector(self, event):
        print(1, self.scatter_selector, self.rect_selector)
        if event.key in ['x', 'X'] and self.rect_selector.active:
            self.rect_selector.set_active(False)
        
        if event.key in ['x', 'X'] and (self.scatter_selector is not None):
            self.fig.canvas.mpl_disconnect(self.scatter_selector)
            self.scatter_selector = None
        
        if event.key in ['a', 'A'] and (self.scatter_selector is None):
            self.select_scatter()
        
        if event.key in ['r', 'R'] and ((self.rect_selector is None) | (not self.rect_selector.active)):
            self.select_rectangle()

    def get_rectangle(self):
        return self.rect_verts, self.rect_area
    
    def overplot_sources(self):
        r"""
        """

    def track_sources(self):
        r"""
        """

    def close(self):
        r"""
        """
        self.fig.close()

    def rectangle_to_source(self):
        r"""
        """
    
    def positions_to_source(self, framesize=50):
        r"""
        """
        coords = np.array(self.positions)
        sources_coords = {'x': coords[:, 0],
                          'y': coords[:,1]}
        sources = SourcesDataFrame(sources_coords, img=self.image)
        return sources

class ImageListViewer(ContrastViewer):

    def __init__(self, wcs=True, cardinal=None, sundirection=True):
        super().__init__(wcs, cardinal, sundirection)
        self.current_index = 0
        self.running = False
        self.thread = None
        self.positions = []
        # self.fig = None
        # self.ax = None

    def __call__(self, images, 
                 ax_im=None, ax_contrast=None, ax_controls=None, fig=None,
                 cardinal_kwargs=None, 
                 label_kwargs=None,
                 return_fig=False,
                 **kwargs):
        if fig is None:
            fig = plt.figure()
            plt.tight_layout()  
        if ax_im is None:
            gs = fig.add_gridspec(4, 3, width_ratios=[10, 0.75, 0.75], height_ratios=[10, 0.75, 0.75, 0.75],)
            if (images[0].wcs is not None) & self.wcs:
                ax_im = fig.add_subplot(gs[0, 0], projection=images[0].wcs)
            else:
                ax_im = fig.add_subplot(gs[0, 0])
                ax_im.set_xlabel('X (px)')
                ax_im.set_ylabel('Y (px)')
            ax_contrast = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
            ax_buttons = [fig.add_subplot(gs[1, 1:]), fig.add_subplot(gs[2, 1:]),
                          fig.add_subplot(gs[3, 1:])]
            ax_controls = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[3, 0])]

        self.images = images
        # self.fig, self.ax = plt.subplots() if fig is None or ax is None else (fig, ax)
        super().__call__(images[0], ax_im=ax_im, ax_sliders=ax_contrast, fig=fig,
                         cardinal_kwargs=cardinal_kwargs, label_kwargs=label_kwargs,
                         **kwargs)
        self.control_buttons(ax_buttons)
        self.control_sliders(ax_controls)
        self.cardinal_kwargs = cardinal_kwargs
        self.label_kwargs = label_kwargs
        self.kwargs = kwargs
        if return_fig:
            return self.images, self.fig, ax_im, ax_controls, ax_contrast

    def control_buttons(self, ax_controls):
        r"""
        """

        ax_play, ax_pause, ax_delete = ax_controls
        self.play_button = Button(ax_play, 'Play')
        self.pause_button = Button(ax_pause, 'Pause')
        self.delete_button = Button(ax_delete, 'Del')

        self.play_button.on_clicked(self.play)
        self.pause_button.on_clicked(self.pause)
        self.delete_button.on_clicked(self.delete_frame)
    
    def control_sliders(self, ax_sliders):

        ax_speed, ax_index = ax_sliders
        self.speed_slider = Slider(ax_speed, 'Speed', 0.1, 2.0, valinit=0.5)
        self.index_slider = Slider(ax_index, 'Index', 0, len(self.images) - 1, 
                                   valstep=1,
                                   valinit=self.current_index, valfmt='%i')

        self.index_slider.on_changed(self.update_image)

        # plt.show()

    def update_image(self, val):
        self.current_index = int(self.index_slider.val)
        self.ax.set_title(self.images[self.current_index].label)
        self.im.set_data(self.images[self.current_index].data)
        self.fig.canvas.draw_idle()

    def blink(self):
        while self.running and len(self.images) > 0:
            self.current_index = (self.current_index + 1) % len(self.images)
            self.index_slider.set_val(self.current_index)
            self.ax.set_title(self.images[self.current_index].label)
            self.im.set_data(self.images[self.current_index].data)
            self.fig.canvas.draw_idle()
            time.sleep(self.speed_slider.val)

    def play(self, event):
        if not self.running:
            self.running = True
            self.thread = Thread(target=self.blink)
            self.thread.start()

    def pause(self, event):
        self.running = False
        if self.thread is not None:
            self.thread.join()

    def delete_frame(self, event):
        if len(self.images) > 0:
            del self.images[self.current_index]
            if self.current_index >= len(self.images):
                self.current_index = 0
            self.index_slider.valmax = len(self.images) - 1
            if len(self.images) > 0:
                self.ax.set_title(self.images[self.current_index].label)
                self.im.set_data(self.images[self.current_index].data)
                # super().__call__(self.images[self.current_index], ax=self.ax, fig=self.fig,
                #           cardinal_kwargs=self.cardinal_kwargs, label_kwargs=self.label_kwargs)
            else:
                self.ax.clear()
                self.fig.canvas.draw_idle()


class CenterFitter(ContrastViewer):
    
    def __init__(self):
        r"""
        """