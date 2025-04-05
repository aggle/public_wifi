"""
Tools for fitting PSFs with MCMC methods
"""
import numpy as np
import pandas as pd
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


class FitStar:
    def __init__(self, stamp, epsf):
        self.epsf = epsf
        self.stamp = stamp
        self.center = misc.get_stamp_center(stamp)
        # self.star = star
        # if use_kklip > 0:
        #     self.stamp = star.results['klip_model'].loc[cat_row_ind, use_kklip]
        # else:
        #     self.stamp = star.cat.loc[cat_row_ind, 'stamp']
        self.sampler = None
        self.labels = ["f", "x", "y"]#, "log(f)"]
        self.initial_values = self.get_default_initial_estimates()
        self.ygrid, self.xgrid = np.mgrid[:self.stamp.shape[0], :self.stamp.shape[1]]
        self.xgrid -= self.center[0]
        self.ygrid -= self.center[1]

    def get_default_initial_estimates(self):
        x0, y0 = 0.0, 0.0 #np.unravel_index(np.argmax(self.stamp), self.stamp.shape)[::-1]
        f0 = self.stamp.sum()
        # log_f = 0.5
        return {'f': f0, 'x': x0, 'y': y0}#, 'log(f)': log_f}

    def log_likelihood(
            self,
            theta : list,
    ) -> float:
        """
        Compute the log likelihood

        Parameters
        ----------
        theta : list
          list of fitting parameters
        stamp : np.ndarray
          2-D image you're trying to fit
        stamp_unc : np.ndarray
          2-D array of uncertainties

        Output
        ------
        ll : float
          the log-likelihood difference between the model and the data
        """
        # [p]rimary [f]lux, [x], [y]
        # log_f is a factor corresponding to underestimated uncertainties
        # fp, xp, yp, log_f = theta
        fp, xp, yp = theta
        # evaluate the epsf for this set of parameters
        model_guess = self.epsf.evaluate(self.xgrid, self.ygrid, fp, xp, yp)
        residual = self.stamp - model_guess
        weight2 = np.sqrt(residual**2)
        ll = -0.5 * np.sum(residual ** 2 / weight2 + np.log(weight2))
        return ll

    def log_prior(self, theta : list) -> float:
        # fp, xp, yp, log_f = theta
        fp, xp, yp = theta
        x0 = self.initial_values['x']
        y0 = self.initial_values['y']
        # f0 = self.initial_values['f']
        # log_f = self.initial_values['log(f)']
        # uniform priors
        fp_prior = (0 < fp < self.stamp.max()*self.stamp.size)
        xp_prior = (x0-0.5 <= xp <= x0+0.5)
        yp_prior = (y0-0.5 <= yp <= y0+0.5)
        # logf_prior = (-10.0 < log_f < 1.0)
        priors = ( fp_prior and xp_prior and yp_prior )# and logf_prior )
        if priors:# and logf_prior:
            return 0.0
        else:
            return -np.inf

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def run_mcmc(self, nwalkers=32):
        """Run the MCMC sampler"""
        x0 = self.initial_values['x']
        y0 = self.initial_values['y']
        f0 = self.initial_values['f']
        # log_f = self.initial_values['log(f)']
        # uniform priors
        init = [f0, x0, y0]#, log_f]
        pos = np.array(init) + 1e-4 * np.random.randn(nwalkers, len(init))
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability,
        )
        sampler.run_mcmc(pos, 5000, progress=True)
        self.sampler = sampler
        return

    def show_chains(self):
        samples = self.sampler.get_chain()
        nchains = len(self.labels)
        fig, axes = plt.subplots(nchains, figsize=(10, 7), sharex=True)
        for i in range(nchains):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(self.labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number");
        return fig

    def show_corner_plot(self):
        flat_samples = self.sampler.get_chain(discard=500, thin=15, flat=True)
        fig = corner.corner(
            flat_samples, labels=self.labels,
        );
        return fig

    def evaluate_parameters(self):
        values = {}
        flat_samples = self.sampler.get_chain(discard=500, thin=15, flat=True)
        for i, label in enumerate(self.labels):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            values[label] = (mcmc[1], *q)
        self.estimates = values

    def print_estimates(self):
        for label, est in self.estimates.items():
            txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(est[0], est[1], est[2], label)
            display(Math(txt))





class FitTwoStars:
    def __init__(
            self,
            stamp,
            epsf,
            params={'f1':0, 'x1': 0, 'y1': 0, 'f2': 0, 'x2': 0, 'y2': 0 },
            star=None,
    ):
        """
        stamp : the data to fit
        epsf : a 2-PSF model!
        """
        self.stamp = stamp
        self.sampler = None
        self.labels = list(params.keys())
        self.initial_values = params.copy()
        self.ygrid, self.xgrid = np.mgrid[:self.stamp.shape[0], :self.stamp.shape[1]]
        self.epsf = epsf
        # shift the coordinates to have 0 at the center
        self.center = misc.get_stamp_center(stamp)
        self.xgrid -= self.center[0]
        self.ygrid -= self.center[1]

    def get_default_initial_estimates(self):
        x0, y0 = 0.0, 0.0 #np.unravel_index(np.argmax(self.stamp), self.stamp.shape)[::-1]
        f0 = self.stamp.sum()
        # log_f = 0.5
        return {'f': f0, 'x': x0, 'y': y0}#, 'log(f)': log_f}

    def log_likelihood(
            self,
            theta : list,
    ) -> float:
        """
        Compute the log likelihood

        Parameters
        ----------
        theta : list
          list of fitting parameters
        stamp : np.ndarray
          2-D image you're trying to fit
        stamp_unc : np.ndarray
          2-D array of uncertainties

        Output
        ------
        ll : float
          the log-likelihood difference between the model and the data
        """
        # [p]rimary [f]lux, [x], [y]
        # log_f is a factor corresponding to underestimated uncertainties
        # fp, xp, yp, log_f = theta
        f1, x1, y1, f2, x2, y2 = theta
        # evaluate the epsf for this set of parameters
        model_guess = self.epsf.evaluate(
            self.xgrid, self.ygrid,
            f1, x1, y1,
            f2, x2, y2
        )
        residual = self.stamp - model_guess
        weight2 = np.sqrt(residual**2)
        ll = -0.5 * np.sum(residual**2)# / weight2)# + np.log(weight2))
        return ll

    def log_prior(self, theta : list) -> float:
        # fp, xp, yp, log_f = theta
        f1, x1, y1, f2, x2, y2 = theta
        f1_0 = self.initial_values['f1']
        x1_0 = self.initial_values['x1']
        y1_0 = self.initial_values['y1']
        f2_0 = self.initial_values['f2']
        x2_0 = self.initial_values['x2']
        y2_0 = self.initial_values['y2']
        # f0 = self.initial_values['f']
        # log_f = self.initial_values['log(f)']
        # uniform priors

        stamp_max = np.ptp(self.stamp)
        f1_prior = (stamp_max < f1 < 1e2*stamp_max)
        # f1_prior = (0.5 * f1_0 < f1 < 1.5 * f1_0)
        x1_prior = (x1_0-0.5 <= x1 <= x1_0+0.5)
        y1_prior = (y1_0-0.5 <= y1 <= y1_0+0.5)
        f2_prior = (0.5*f1 <= f2 < f1)
        x2_prior = (x2_0-0.5 <= x2 <= x2_0+0.5)
        y2_prior = (y2_0-0.5 <= y2 <= y2_0+0.5)
        # logf_prior = (-10.0 < log_f < 1.0)
        priors = ( f1_prior and x1_prior and y1_prior and f2_prior and x2_prior and y2_prior)# and logf_prior )
        if priors:# and logf_prior:
            return 0.0
        else:
            return -np.inf

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)

    def run_mcmc(self, nwalkers=32):
        """Run the MCMC sampler"""
        x1 = self.initial_values['x1']
        y1 = self.initial_values['y1']
        f1 = self.initial_values['f1']
        x2 = self.initial_values['x2']
        y2 = self.initial_values['y2']
        f2 = self.initial_values['f2']
        # log_f = self.initial_values['log(f)']
        # uniform priors
        init = [f1, x1, y1, f2, x2, y2]#, log_f]
        pos = np.array(init) + 1e-4 * np.random.randn(nwalkers, len(init))
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability,
        )
        sampler.run_mcmc(pos, 5000, progress=True)
        self.sampler = sampler
        return sampler


    def evaluate_parameters(self):
        values = {}
        flat_samples = self.sampler.get_chain(discard=500, thin=15, flat=True)
        for i, label in enumerate(self.labels):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            values[label] = (mcmc[1], *q)
        self.estimates = values
        # generate a model from the parameter estimates
        self.best_model = self.generate_image_from_parameters()

    def print_estimates(self):
        for label, est in self.estimates.items():
            txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(est[0], est[1], est[2], label)
            display(Math(txt))

    def generate_image_from_parameters(self):
        theta = [i[0] for i in self.estimates.values()]
        model = self.epsf.evaluate(self.xgrid, self.ygrid, *theta)
        return model

    def compute_seppa(self, wcs):
        primary_xy = misc.center_to_ll_coords(self.stamp.shape[0], (self.estimates['x1'][0], self.estimates['y1'][0]))
        companion_xy = misc.center_to_ll_coords(self.stamp.shape[0], (self.estimates['x2'][0], self.estimates['y2'][0]))
        prim_sky = wcs.pixel_to_world(*primary_xy)
        comp_sky = wcs.pixel_to_world(*companion_xy)
        sep = prim_sky.separation(comp_sky).to("mas")
        pa = prim_sky.position_angle(comp_sky).to("deg")
        return sep, pa


    def show_chains(self):
        samples = self.sampler.get_chain()
        nchains = len(self.labels)
        fig, axes = plt.subplots(nchains, figsize=(10, 7), sharex=True)
        for i in range(nchains):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(self.labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number");
        return fig

    def show_corner_plot(self):
        flat_samples = self.sampler.get_chain(discard=500, thin=15, flat=True)
        fig = corner.corner(
            flat_samples, labels=self.labels,
        );
        return fig

    def plot_data_model(self):

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), layout='constrained')

        plotyx = np.mgrid[:self.stamp.shape[0]+1, :self.stamp.shape[1]+1] - 0.5 - misc.get_stamp_center(self.stamp)[:, None, None]
        vmin, vmax = np.nanquantile(np.stack([self.stamp, self.best_model]), [0, 1])

        ax = axes[0]
        ax.set_title("Data")
        imax = ax.pcolor(
            plotyx[1], plotyx[0], self.stamp,
            vmin=vmin, vmax=vmax
        )
        fig.colorbar(imax, ax=ax, orientation='horizontal')
        ax = axes[1]
        ax.set_title("Model")
        imax = ax.pcolor(
            plotyx[1], plotyx[0], self.best_model,
            vmin=vmin, vmax=vmax
        )
        fig.colorbar(imax, ax=ax, orientation='horizontal')

        ax = axes[2]
        ax.set_title("(Data - Model)")
        residual = (self.stamp - self.best_model)
        # vmin, vmax = np.nanquantile(residual, [0.05, 0.95])
        imax = ax.pcolor(
            plotyx[1], plotyx[0], residual,
            norm=mpl.colors.TwoSlopeNorm(vcenter=0),
        )
        fig.colorbar(imax, ax=ax, orientation='horizontal')

        for ax in axes:
            ax.errorbar(
                self.estimates['x1'][0], self.estimates['y1'][0],
                xerr=np.array(self.estimates['x1'])[1:, None],
                yerr=np.array(self.estimates['y1'])[1:, None],
                marker='*',
                c='k'
            )
            ax.errorbar(
                self.estimates['x2'][0], self.estimates['y2'][0],
                xerr=np.array(self.estimates['x2'])[1:, None],
                yerr=np.array(self.estimates['y2'])[1:, None],
                marker='+',
                c='k'
            )
            ax.set_aspect("equal")

        return fig

def make_forward_modeled_psf(
        epsf : pupsf.image_models.ImagePSF,
        x : float,
        y : float,
        klip_basis : pd.Series,
        flux : float = 1.0,
        kklip : int | None = None
) -> np.ndarray :
    """
    Forward model a companion PSF through the KLIP basis. Return a KLIP-modified PSF.

    Parameters
    ----------
    epsf : an oversampled effective PSF object from photutils
    x, y : float
      the (col, row) location of the PSF
    klip_basis : pd.Series
      the KLIP basis vectors, indexed by order
    flux : float = 1
      normalize the flux of the generated PSF to this value
    width : int | None = None
      if None, return then entire generated PSF. Otherwise, return a stamp of (width, width) shape
    kklip : int | None = None
      If given, return only the psf for this kklip correction. If None, return all KKlips

    Output
    ------
    fm_companion : a stamp of the modified PSF

    """
    stamp_shape = misc.get_stamp_shape(klip_basis)
    ygrid, xgrid = np.mgrid[:stamp_shape[0], :stamp_shape[1]]
    psf = epsf.evaluate(xgrid, ygrid, flux, x, y)
    # subtract mean before applying klip basis
    psf = psf - psf.mean()
    # compute the projection of the psf onto each kb mode
    kb_coeffs = np.cumsum(
        klip_basis.apply(lambda kb: kb * np.dot(kb.ravel(), psf.ravel()))
    )
    psf_corrected = kb_coeffs.apply(lambda coeff: psf - coeff)
    if kklip is not None:
        psf_corrected = psf_corrected.loc[kklip]
    return psf_corrected

