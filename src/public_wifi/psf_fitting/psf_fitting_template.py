"""
Tools for fitting PSFs with MCMC methods
"""
import abc
import numpy as np
import pandas as pd

from scipy.optimize import minimize

from astropy.coordinates import SkyCoord
from photutils import psf as pupsf

import matplotlib as mpl
from matplotlib import pyplot as plt
import emcee
import corner

from IPython.display import display, Math

from public_wifi import misc

def star2epsf(star, cat_row_ind : int) -> pupsf.epsf_stars.EPSFStar:
    """Convert a star to an EPSF object"""
    stamp = star.cat.loc[cat_row_ind, 'cutout'].data
    bgnd = star.cat.loc[cat_row_ind, 'bgnd'][0]
    # normalize the stamp to max 1
    # weights = star.cat.loc[cat_row_ind, 'cutout_err'].data
    weights = None
    # weights = np.sqrt(stamp - stamp.min())
    center = misc.get_stamp_center(stamp)[::-1]
    epsf = pupsf.EPSFStar(
        stamp - bgnd,
        weights=weights,
        id_label=star.star_id,
        cutout_center=center
    )
    return epsf

def construct_epsf(
        stars : pd.Series,
        cat_row_ind : int,
        oversampling : int = 3
) -> pupsf.epsf_stars.EPSFStars :
    """
    Construct an EPSF from a pd.Series of Stars objects.
    uses all the stars provided, so filtering must be done before passing.

    Parameters
    ----------
    stars : pd.Series
      a pd.Series of starclass.Star objects
    cat_row_ind : int
      which row of the catalog to take from

    Output
    ------
    epsfs : an EPSFStars object
    """

    epsfs = pupsf.EPSFStars([star2epsf(star, cat_row_ind) for star in stars])
    epsf_builder = pupsf.EPSFBuilder(
        oversampling = oversampling,
        maxiters=10,
        progress_bar=False
    )

    epsf, fitted_stars = epsf_builder(epsfs)
    return epsf


class FitPSFTemplate(abc.ABC):
    def __init__(
            self, 
            star, 
            epsf, 
            cat_row : int = 0,
            param_labels=[],
    ) -> None:
        # self.star = star
        self.epsf = epsf
        self._img = star.cat.loc[cat_row, 'stamp']
        self.err = star.cat.loc[cat_row, 'cutout_err'].data
        self.shape = self.img.shape
        self.cat_row = star.cat.loc[cat_row]

        self.center = misc.get_stamp_center(self.img)
        # self.mask = self.make_fitting_mask(self.img)
        self.ygrid, self.xgrid = np.mgrid[:self.img.shape[0], :self.img.shape[1]]
        self.xgrid -= self.center[0]
        self.ygrid -= self.center[1]


        self.sampler = None
        # labels
        self.param_labels = param_labels  #['f', 'x1', 'y1']
        # self.initial_values =  np.zeros(len(self.param_labels), dtype=float)
        self.initial_values = self.set_initial_values()
        return None

    @abc.abstractmethod
    def set_initial_values(self) -> np.ndarray:
        pass
    @abc.abstractmethod
    def generate_model(self, theta) -> np.ndarray:
        pass

    @abc.abstractmethod
    def log_prior(self, theta) -> float:
        pass

    def log_likelihood(self, theta) -> float:
        """Simple chi2 likelihood function"""
        img = self.img
        model = self.generate_model(theta)
        # chi2
        unc = np.sqrt(img - img.min())
        ll = -0.5 * np.sum(
            ( (img - model)**2 ) #/ ( unc**2 ) + np.log(2*np.pi*unc**2)
        )
        return ll


    @property
    def img(self):
        return self._img
    @img.setter
    def img(self, newval):
        self._img = newval
        # self.mask = self.make_fitting_mask(newval)

    def make_fitting_mask(self, img):
        # only include the brightest 25% of pixels
        # thresh = np.nanquantile(img, 0.85)
        # mask = img <= thresh
        mask = np.ones_like(self.img, dtype=bool)
        rad = 3
        center = self.center
        mask[center[1]-rad:center[1]+rad+1, center[0]-rad:center[0]+rad+1] = False
        return mask

    def guess_lstq_opt(self, theta, return_full_result=False):
        nll = lambda *args: -self.log_likelihood(*args)
        init = theta#self.initial_values
        result = minimize(
            nll,
            init,
            bounds=[
                (0, self.img.max()*self.img.size),
                (-2, 2),
                (-2, 2)
            ]
        )
        if return_full_result:
            return result
        return result.x

    def log_probability(self, theta) -> float:
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)
        
    def run_mcmc(self, nwalkers=32, nsteps=5000, threads=1) -> None:
        initial_values = self.initial_values
        pos = initial_values + 1e-4 * np.random.randn(32, len(self.initial_values))
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            self.log_probability,
            threads=threads,
        )
        sampler.run_mcmc(pos, nsteps, progress=True)
        self.sampler = sampler
        values = self.evaluate_parameters()
        return

    def evaluate_parameters(self, discard=None):
        values, uncertainties = [], []
        if discard is None:
            discard = int(np.floor(len(self.sampler.get_chain())*0.1))
        flat_samples = self.sampler.get_chain(discard=discard, thin=15, flat=True)
        for i, label in enumerate(self.param_labels):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            values.append(mcmc[1])
            uncertainties.append(q)
        self.estimates = values
        self.uncertainties = uncertainties
        return
        
    ### Plotting and printing ###
    
    def show_chains(self):
        samples = self.sampler.get_chain()
        nchains = len(self.param_labels)
        fig, axes = plt.subplots(nchains, figsize=(10, 7), sharex=True)
        for i in range(nchains):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(self.param_labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number");
        return fig

    def show_corner_plot(self, discard=None):
        # discard=500
        # if len(self.sampler.get_chain()) <= 500:
        # discard the first 20% by default
        if discard is None:
            discard = int(np.floor(len(self.sampler.get_chain())*0.1))
        flat_samples = self.sampler.get_chain(discard=discard, thin=15, flat=True)
        fig = corner.corner(
            flat_samples, labels=self.param_labels,
        );
        return fig
        
    def compare_model(self, theta):
        model = self.generate_model(theta)
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        ax = axes[0, 0]
        ax.set_title("Data")
        imax = ax.imshow(self.img)
        fig.colorbar(imax, ax=ax, orientation='horizontal')
        ax = axes[0, 1]
        ax.set_title("Model")
        imax = ax.imshow(model)
        fig.colorbar(imax, ax=ax, orientation='horizontal')
        ax = axes[1, 0]
        ax.set_title("Data - Model")
        residual = self.img - model
        imax = ax.imshow(residual, norm=mpl.colors.CenteredNorm()) #)
        fig.colorbar(imax, ax=ax, orientation='horizontal')
        ax = axes[1, 1]
        ax.set_title("(Data - Model) / Data")
        residual = (self.img - model)/self.img
        vmin, vmax = np.nanquantile(residual, [0.05, 0.95])
        imax = ax.imshow(residual, norm=mpl.colors.SymLogNorm(linthresh=0.1)) # vmin=vmin, vmax=vmax) # 
        fig.colorbar(imax, ax=ax, orientation='horizontal')

        for ax in axes.flat:
            ax.scatter(*self.center, marker='x', c='w')

        return fig

    
    def print_estimates(self):
        self.evaluate_parameters()
        for i, label in enumerate(self.param_labels):
            val = self.estimates[i]
            unc_lo, unc_hi = self.uncertainties[i]
            txt = "\mathrm{{{0}}} = {1:.3f}_{{-{2:.3f}}}^{{{3:.3f}}}"
            txt = txt.format(label, val, unc_lo, unc_hi)
            display(Math(txt))
        return
        
    def show_all_results(self) -> tuple:
        values = self.evaluate_parameters()
        self.print_estimates()
        chain_fig = self.show_chains()
        corner_fig = self.show_corner_plot()
        comp_fig = self.compare_model(self.estimates)
        return (chain_fig, corner_fig, comp_fig)
