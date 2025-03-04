"""
Tools for fitting PSFs with MCMC methods
"""
import numpy as np
import pandas as pd
from photutils import psf as pupsf

from matplotlib import pyplot as plt
import emcee
import corner

from public_wifi import misc

def star2epsf(star, cat_row_ind : int) -> pupsf.epsf_stars.EPSFStar:
    """Convert a star to an EPSF object"""
    stamp = star.cat.loc[cat_row_ind, 'cutout'].data
    # weights = star.cat.loc[cat_row_ind, 'cutout_err'].data
    weights = np.sqrt(stamp - stamp.min())
    center = misc.get_stamp_center(stamp)[::-1]
    epsf = pupsf.EPSFStar(
        stamp,
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
    def __init__(self, star, epsf, cat_row_ind, use_kklip : int = 0):
        self.epsf = epsf
        self.star = star
        if use_kklip > 0:
            self.stamp = star.results['klip_model'].loc[cat_row_ind, use_kklip]
        else:
            self.stamp = star.cat.loc[cat_row_ind, 'stamp']
        self.stamp_unc = np.sqrt(self.stamp - self.stamp.min())
        self.sampler = None
        self.labels = ["f", "x", "y", "log(f)"]
        self.initial_values = self.get_default_initial_estimates()

    def get_default_initial_estimates(self):
        x0, y0 = misc.get_stamp_center(self.stamp)
        f0 = self.stamp.max()
        log_f = 0.5
        return {'f': f0, 'x': x0, 'y': y0, 'log(f)': log_f}

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
        stamp_shape = self.stamp.shape
        ycoords, xcoords = np.mgrid[:stamp_shape[0], :stamp_shape[1]]
        # [p]rimary [f]lux, [x], [y]
        # log_f is a factor corresponding to underestimated uncertainties
        fp, xp, yp, log_f = theta
        # evaluate the epsf for this set of parameters
        model_guess = self.epsf.evaluate(xcoords, ycoords, fp, xp, yp)
        sigma2 = self.stamp_unc**2 + model_guess**2 * np.exp(2*log_f)
        ll = -0.5 * np.sum((self.stamp - model_guess) ** 2 / sigma2 + np.log(sigma2))
        return ll

    def log_prior(self, theta : list) -> float:
        fp, xp, yp, log_f = theta
        x0 = self.initial_values['x']
        y0 = self.initial_values['y']
        f0 = self.initial_values['f']
        # uniform priors
        fp_prior = (0 < fp <= f0*self.stamp.size)
        xp_prior = (x0-0.5 < xp < x0+0.5)
        yp_prior = (y0-0.5 < yp < y0+0.5)
        if fp_prior and xp_prior and yp_prior:
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
        log_f = self.initial_values['log(f)']
        # uniform priors
        init = [f0, x0, y0, log_f]
        pos = np.array(init)+ 1e-4 * np.random.randn(nwalkers, len(init))
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability,
        )
        sampler.run_mcmc(pos, 5000, progress=True)
        self.sampler = sampler
        return

    def show_chains(self):
        fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
        samples = self.sampler.get_chain()
        for i in range(4):
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
