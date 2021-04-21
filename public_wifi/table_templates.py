"""
This file contains the templates for all the tables.
"""

import pandas as import pd

# stars table
columns = {
    'star_id': object,
    'u_mast': np.float32,
    'v_mast': np.float32
    'star_phot_F1': np.float32,
    'star_phot_e_F1': np.float32,
    'star_phot_F2': np.float32,
    'star_phot_e_F2': np.float32,
    'star_mag_F1': np.float32,
    'star_mag_e_F1': np.float32,
    'star_mag_F2': np.float32,
    'star_mag_e_F2': np.float32,
    'clust_memb': bool,
}
stars_table = pd.DataFrame(columns=columns.keys())

# point sources table
columns = {
    'ps_id': object,
    'ps_star_id': object,
    'ps_exp_id': object,
    'ps_filt_id': object,
    'ps_epoch_id': object,
    'ps_x_exp,': np.float32,
    'ps_y_exp': np.float32,
    'ps_u_mast': np.float32,
    'ps_v_mast': np.float32,
    'ps_phot': np.float32,
    'ps_phot_e': np.float32,
    'ps_mag': np.float32,
    'ps_mag_e': np.float32,
    'ps_psf_fit': np.float32,
}
point_sources_table = pd.DataFrame(column=columns.keys())

# stamps table
columns = {
    'stamp_id': object,
    'stamp_ps_id': object,
    'stamp_star_id': object,
    'stamp_x_cent': np.float32,
    'stamp_y_cent': np.float32,
    'stamp_path': object,
    'stamp_ref_flag': bool
}
stamps_table = pd.DataFrame(columns=columns.keys())

# stamp data quality table
columns = {
    'sdq_stamp_id': object,
    'sdq_flag': object,
    'sdq_flag_value': object
}
stamps_dq_table = pd.DataFrame(columns=columns.keys())

# companion_status
columns = {
    'comp_star_id': object,
    'companion_flag': np.int16
}
companion_status_table = pd.DataFrame(columns=columns.keys())

#################
# Lookup Tables #
#################
data = [
    (0, 'single'),
    (1, 'visual'),
    (2, 'faint')
]
lookup_companion_flag = pd.DataFrame(data, columns=['flag', 'value'])

