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

from public_wifi.psf_fitting import psf_fitting_template as pft

uniform_priors = {
    'f': (0., np.inf),
    'x1': (-1. , +1.),
    'y1': (-1. , +1.),
    'c': (0., 1.),
    'x2': (-4. , -2.),
    'y2': (0. , +2.),
}


class FitPSF(pft.FitPSFTemplate):
    def __init__(
            self, 
            star, 
            epsf, 
            cat_row : int = 0,
    ) -> None:
        param_labels = ['f','x','y']
        super().__init__(star, epsf, cat_row, param_labels)
        return

    def set_initial_values(self) -> None:
        initial_values = np.array([
            self.img.max()*9,
            0, 0
        ])
        return initial_values
                                  
    def generate_model(self, theta) -> np.ndarray:
        # f, x1, y1, *_ = theta
        # model = make_1psf_model(theta, self.epsf, self.shape)
        fp, xp, yp = theta
        # evaluate the epsf for this set of parameters
        model = self.epsf.evaluate(self.xgrid, self.ygrid, fp, xp, yp)
        return model

    def log_prior(self, theta) -> float:
        # xc, yc = misc.get_stamp_center(self.img)
        f, x1, y1, *_ = theta
        prior = {
            'f': (0 < f <= self.img.max() * self.img.size),
            'x1': (-2 <= x1 <= +2),
            'y1': (-2 <= y1 <= +2),
        }
        prob = 0
        if not all(prior.values()):
            prob = -np.inf
        return prob
        
    def log_likelihood(self, theta) -> float:
        """Simple chi2 likelihood function"""
        img = self.img#[self.mask]
        model = self.generate_model(theta)#[self.mask]
        # chi2
        unc = np.sqrt(img - img.min())
        ll = -0.5 * np.sum(
            ( (img - model)**2 ) #/ ( unc**2 ) + np.log(2*np.pi*unc**2)
        )
        return ll


class FitPSF_flux(pft.FitPSFTemplate):
    def __init__(
            self, 
            star, 
            epsf, 
            cat_row : int = 0,
            x0=0., y0=0.
    ) -> None:
        param_labels = ['f']
        super().__init__(star, epsf, cat_row, param_labels)
        self.x0 = x0
        self.y0 = y0

    def set_initial_values(self) -> np.ndarray:
        initial_values = np.array([
            self.img.max()*np.sqrt(self.img.size)
        ])
        return initial_values

    def generate_model(self, theta, x=None, y=None) -> np.ndarray:
        # f, x1, y1, *_ = theta
        # model = make_1psf_model(theta, self.epsf, self.shape)
        f, *_ = theta
        x = self.x0
        y = self.y0
        # evaluate the epsf for this set of parameters
        model = self.epsf.evaluate(self.xgrid, self.ygrid, f, x, y)
        return model

    def log_prior(self, theta) -> float:
        # xc, yc = misc.get_stamp_center(self.img)
        f, *_ = theta
        prior = {
            'f': (0 < f <= np.ptp(self.img) * self.img.size),
        }
        prob = 0
        if not all(prior.values()):
            prob = -np.inf
        return prob

class FitPSF_position(pft.FitPSFTemplate):
    def __init__(
            self, 
            star, 
            epsf, 
            cat_row : int = 0,
            f0 = 1.
    ) -> None:
        param_labels = ['x1','y1']
        super().__init__(star, epsf, cat_row, param_labels)
        self.f0 = f0

    def set_initial_values(self) -> np.ndarray:
        initial_values = np.array([0., 0.])
        return initial_values

    def generate_model(self, theta) -> np.ndarray:
        # f, x1, y1, *_ = theta
        # model = make_1psf_model(theta, self.epsf, self.shape)
        x1, y1, *_ = theta
        # evaluate the epsf for this set of parameters
        model = self.epsf.evaluate(self.xgrid, self.ygrid, self.f0, x1, y1)
        return model

    def log_prior(self, theta) -> float:
        # xc, yc = misc.get_stamp_center(self.img)
        x1, y1, *_ = theta
        prior = {
            'x1': (-2 <= x1 <= +2),
            'y1': (-2 <= y1 <= +2),
        }
        prob = 0
        if not all(prior.values()):
            prob = -np.inf
        return prob


class FitTwoPSFs(pft.FitPSFTemplate):
    def __init__(
            self, 
            star, 
            epsf, 
            cat_row : int = 0,
    ) -> None:
        param_labels = ['f','x1','y1', 'c', 'x2', 'y2']
        super().__init__(star, epsf, cat_row, param_labels)

        return

    def set_initial_values(self) -> None:
        initial_values = np.array([
            self.img.max()*9, 0, 0,
            1., -3., 1.,
        ])
        return initial_values
                                  
    def generate_model(self, theta) -> np.ndarray:
        f, x1, y1, c, x2, y2 = theta
        # f = theta['f']
        # x1 = theta['x1']
        # y1 = theta['y1']
        # c = theta['c']
        # x2 = theta['x2']
        # y2 = theta['y2']
        # evaluate the epsf for this set of parameters
        model1 = self.epsf.evaluate(self.xgrid, self.ygrid, f, x1, y1)
        model2 = self.epsf.evaluate(self.xgrid, self.ygrid, f*c, x2, y2)
        model = model1 + model2
        return model

    def log_prior(self, theta) -> float:
        # xc, yc = misc.get_stamp_center(self.img)
        f, x1, y1, c, x2, y2, *_ = theta
        prior = {
            'f':  (0 < f <= self.img.max() * self.img.size),
            'x1': (-2. <= x1 <= +2.),
            'y1': (-2. <= y1 <= +2.),
            'c':  (0. <= c <= 1.),
            'x2': (-2. <= x2 <= +2.),
            'y2': (-2. <= y2 <= +2.),
        }
        prob = 0
        if not all(prior.values()):
            prob = -np.inf
        return prob
        
    def log_likelihood(self, theta) -> float:
        """Simple chi2 likelihood function"""
        img = self.img#[self.mask]
        model = self.generate_model(theta)#[self.mask]
        # chi2
        unc = np.sqrt(img - img.min())
        ll = -0.5 * np.sum(
            ( (img - model)**2 ) #/ ( unc**2 ) + np.log(2*np.pi*unc**2)
        )
        return ll

    def guess_lstq_opt(self, theta):
        nll = lambda *args: -self.log_likelihood(*args)
        init = theta#self.initial_values
        result = pft.minimize(
            nll,
            init,
            bounds=[
                (0, self.img.max()*self.img.size),
                (-0.5, 0.5),
                (-0.5, 0.5),
                (0, 1.),
                (-4, -2),
                (0, 2),
            ]
        )
        return result.x
