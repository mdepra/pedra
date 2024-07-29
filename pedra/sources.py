import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats

from .viewer import SourceViewer, ImageViewer
from .util import multiple_moffats2D, multiple_gaussians2D
from .core import Image
        
def detect_sources(img, fwhm=3, threshold=5, sigma=3, sort_by='flux', nsources=None): #-> not working properly
    r"""
    """
    mean, median, std = sigma_clipped_stats(img.data, sigma=sigma)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)  
    sources = daofind(img.data - median)
    sources.sort(sort_by, reverse=True)
    sources = sources.to_pandas()
    if nsources is not None:
        sources = sources.iloc[:nsources]
    if img.wcs is not None:
        pix = np.array([sources['xcentroid'], sources['ycentroid']]).T
        sky = np.array(img.wcs.pixel_to_world_values(pix)).T
        sources['ra'] = sky[0]
        sources['dec'] = sky[1]
    return sources


class SourcesDataFrame(pd.DataFrame):
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, img=None):
        # assert 'x' in columns, "Dataframe must have x and y columns"
        super().__init__(data, index, columns, dtype, copy)
        self.img = img

    @property
    def _constructor(self):
        return SourcesDataFrame

    @property
    def _constructor_sliced(self):
        return lambda *args, **kwargs: Source(*args, **kwargs, img=self.img)

    def get_source(self, idx, frame_size=[50, 50], offset=[0,0]):
        result = self.loc[idx]
        if self.img is None:
            img = self.img
        else:
            trim_lim = self.define_box(result['x'], result['y'], frame_size=frame_size, offset=offset)
            img = self.img.trim(trim_lim)
        return Source(result, img=img, dataframe=self, index=idx, offset=offset)
    
    def get_sources(self, sourcelist, frame_size=[50, 50], offset=[0,0]):
        sources = []
        for idx in sourcelist:
            sources.append(self.get_source(idx, 
                                           frame_size=frame_size,
                                           offset=offset))
        return sources
    
    def update_photocenter(self, sources, frame_size=[50, 50],
                           fit_type='gaussian', n=1,
                           initial_guess=None, update_coords=True):
        r"""
        """
        sources = np.atleast_1d(sources)
        result = []
        for s in sources:
            s = self.get_source(s, frame_size=frame_size)
            s.fit_photocenter(fit_type=fit_type, n=n,
                              initial_guess=initial_guess, update_coords=update_coords)
            result.append(s)
        return result

        
    @staticmethod
    def define_box(x, y, frame_size=[50, 50], offset=[0,0]):
        r"""
        """
        size_x, size_y = frame_size
        off_x, off_y = offset

        lower_x_lim = int(max(x - size_x // 2 + off_x, 0))
        upper_x_lim = int(min(x + size_x // 2 + off_x, float('inf')))

        lower_y_lim = int(max(y - size_y // 2 + off_y, 0))
        upper_y_lim = int(min(y + size_y // 2 + off_y, float('inf')))

        return [[lower_x_lim, upper_x_lim], [lower_y_lim, upper_y_lim]]

class Source(pd.Series):
    r"""
    """

    def __init__(self, *args, img=None, dataframe=None, index=None, offset=[0,0], **kwargs):
        super().__init__(*args, **kwargs)
        self.img = img
        self.dataframe = dataframe
        self._index = index
        self.offset = offset
        # placeholders
        self.fit = None
        self.fit_integral = None
        self._fit_type = None
        self._fit_n = None
        self._fit_params = None
        self._original_center = [self.get('x', None), self.get('y', None)]

    @property
    def _constructor(self):
        return Source

    @property
    def _constructor_expanddim(self):
        return SourcesDataFrame
    
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.dataframe is not None and self._index is not None:
            self.dataframe.loc[self._index, key] = value
    
    def fit_photocenter(self, fit_type='gaussian', n=1,
                        initial_guess=None, update_coords=True,
                        insert_integral=True):
        r"""
        """
        img = self.img
        img.mask_nan()
        ny, nx = img.data.shape
        x = np.arange(nx)
        y = np.arange(ny)
        x, y = np.meshgrid(x, y)
        # Initial guess for parameters
        if initial_guess is None:
            initial_guess = []
            for i in range(n):
                if fit_type.strip().lower() == 'gaussian':
                    initial_guess += [img.data.max(), nx // 2, nx // 2, ny / 4, nx / 4, 0, img.data.min()]
                    # initial_guess += [data.max(), ncols // 2, nrows // 2, ncols / 4, nrows / 4, data.min()]
                elif fit_type.strip().lower() == 'moffat':
                    initial_guess.extend([img.data.max() / n, nx // 2, ny // 2, 1, 2, 0, img.data.min()])
                    # initial_guess += [img.data.max(), nx // 2, nx // 2, ny / 4, nx / 4, img.data.min()]

        z = np.ravel(img.data)
        if fit_type.strip().lower() == 'gaussian':
            func = multiple_gaussians2D
        elif fit_type.strip().lower() == 'moffat':
            func = multiple_moffats2D      

        try:
            self._fit_params, pcov = opt.curve_fit(func, (x, y), z, p0=initial_guess)
        except RuntimeError:
            self._fit_params = np.zeros_like(initial_guess)  # Return zeros if fitting fails
        self.fit = img.copy()
        self.fit.data, self.fit_integral = func((x, y), *self._fit_params, return_flat=False, return_integral=True)
        self._fit_type = fit_type
        self._fit_n = n
        if update_coords:
            self.loc['x'] =  self._original_center[0] - nx//2 + self._fit_params[1] + self.offset[0]
            self.loc['y'] =  self._original_center[1] - ny//2 + self._fit_params[2] + self.offset[1]
            self.__setitem__('x', self.loc['x'])
            self.__setitem__('y', self.loc['y'])
        # if compute_integral:

        return self.fit, self._fit_params

    def view3D(self, ax=None, fig=None,
            show=False, savefig=False, figdir='.',
            label_kwargs=None,
            **kwargs):
        r''' Display image

        Parameters
        ----------
        fax (Optional): None, matplotlib.axes
            If desired to subplot image in a figure. Default is 'None', which
            will open a new plt.figure()

        show (Optional): boolean
            True if want to plt.show(). Default is True.

        savefig (Optional): boolean
            True if want to save image.

        figdir (Optional): str
            Only needed if savefig=True. Directory for saving image.
            The file basename will be the same name of the image fitsfile.
            The image extention is .png.

        **kwargs: matplotlib kwargs

        ''' 
        viewer = SourceViewer()
        fig, ax = viewer(self, ax=ax, fig=fig,
                            label_kwargs=label_kwargs,
                            **kwargs)

        if savefig:
            plt.savefig(figdir+self.name+'.png')
        if show:
            plt.show()
        return fig, ax
    
    def view(self, ax=None, fig=None,
             show=False, savefig=False, figdir='.',
             wcs=False, 
             cardinal=None,
             sundirection=True,
             cardinal_kwargs=None, 
             label_kwargs=None,
              **kwargs):
            r''' Display image

            Parameters
            ----------
            fax (Optional): None, matplotlib.axes
                If desired to subplot image in a figure. Default is 'None', which
                will open a new plt.figure()

            show (Optional): boolean
                True if want to plt.show(). Default is True.

            savefig (Optional): boolean
                True if want to save image.

            figdir (Optional): str
                Only needed if savefig=True. Directory for saving image.
                The file basename will be the same name of the image fitsfile.
                The image extention is .png.

            **kwargs: matplotlib kwargs

            ''' 
            viewer = ImageViewer(wcs=wcs, 
                                 cardinal=cardinal,
                                 sundirection=sundirection)
            fig, ax = viewer(self.img, ax=ax, fig=fig,
                             cardinal_kwargs=cardinal_kwargs, 
                             label_kwargs=label_kwargs,
                             **kwargs)

            if savefig:
                plt.savefig(savefig)
            if show:
                plt.show()
            return fig, ax
    
    def plot_fit(self, ax=None, fig=None,
                 show=False, savefig=False, figdir='.',
                 label_kwargs=None,
                **kwargs):
        r''' Display image

        Parameters
        ----------
        fax (Optional): None, matplotlib.axes
            If desired to subplot image in a figure. Default is 'None', which
            will open a new plt.figure()

        show (Optional): boolean
            True if want to plt.show(). Default is True.

        savefig (Optional): boolean
            True if want to save image.

        figdir (Optional): str
            Only needed if savefig=True. Directory for saving image.
            The file basename will be the same name of the image fitsfile.
            The image extention is .png.

        **kwargs: matplotlib kwargs

        ''' 
        viewer = SourceViewer()
        fig, ax_m, ax_r = viewer.view_fit(self, ax=ax, fig=fig,
                                  label_kwargs=label_kwargs,
                                  **kwargs)

        if savefig:
            plt.savefig(savefig)
        if show:
            plt.show()
        return fig, ax_m, ax_r

    def plot_fit3D(self, ax=None, fig=None,
                 show=False, savefig=False, figdir='.',
                 label_kwargs=None,
                **kwargs):
        r''' Display image

        Parameters
        ----------
        fax (Optional): None, matplotlib.axes
            If desired to subplot image in a figure. Default is 'None', which
            will open a new plt.figure()

        show (Optional): boolean
            True if want to plt.show(). Default is True.

        savefig (Optional): boolean
            True if want to save image.

        figdir (Optional): str
            Only needed if savefig=True. Directory for saving image.
            The file basename will be the same name of the image fitsfile.
            The image extention is .png.

        **kwargs: matplotlib kwargs

        ''' 
        viewer = SourceViewer()
        fig, ax = viewer.view_fit3D(self, ax=ax, fig=fig,
                                  label_kwargs=label_kwargs,
                                  **kwargs)

        if savefig:
            plt.savefig(savefig)
        if show:
            plt.show()
        return fig, ax
    
    def view_residual(self, ax=None, fig=None,
                      show=False, savefig=False, figdir='.',
                      wcs=False, 
                      cardinal=None,
                      sundirection=True,
                      cardinal_kwargs=None, 
                      label_kwargs=None,
                      **kwargs):
            r''' Display image

            Parameters
            ----------
            fax (Optional): None, matplotlib.axes
                If desired to subplot image in a figure. Default is 'None', which
                will open a new plt.figure()

            show (Optional): boolean
                True if want to plt.show(). Default is True.

            savefig (Optional): boolean
                True if want to save image.

            figdir (Optional): str
                Only needed if savefig=True. Directory for saving image.
                The file basename will be the same name of the image fitsfile.
                The image extention is .png.

            **kwargs: matplotlib kwargs

            ''' 
            viewer = ImageViewer(wcs=wcs, 
                                 cardinal=cardinal,
                                 sundirection=sundirection)
            res = self.img - self.fit
            fig, ax = viewer(res, ax=ax, fig=fig,
                             cardinal_kwargs=cardinal_kwargs, 
                             label_kwargs=label_kwargs,
                             **kwargs)

            if savefig:
                plt.savefig(figdir+self.name+'.png')
            if show:
                plt.show()
            return fig, ax