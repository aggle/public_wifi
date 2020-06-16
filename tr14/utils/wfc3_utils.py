"""
This file contains data particular to handling WFC3 data, such as error codes
"""

import numpy as np
import pandas as pd

"""
Description of the different Data quality flags.
Source:
https://hst-docs.stsci.edu/wfc3dhb/chapter-2-wfc3-data-structure/2-2-wfc3-file-structure#id-2.2WFC3FileStructure-2.2.3ContentsofIndividualArrays
"""
dq_flags_descr = {0: "OK", # 0
                  1: "Reed Solomon decoding error", # 1
                  2: "Data missing, replaced by fill value", # 2
                  4: "Bad detector pixel", # 4
                  8: "Deviant zero read (bias) value", # 5
                  16: "Hot pixel", # 6
                  32: "Unstable response", # 7
                  64: "Warm pixel", # 8
                  128: "Bad reference pixel", # 9
                  256: "Full well saturation", # 10
                  512: "Bad or uncertain flat value", # 11
                  1024: "(Reserved)", # 11
                  2048: "Signal in zero read", # 12
                  4096: "Cosmic ray detected by AstroDrizzle", # 13
                  8192: "Cosmic detected during calwf3 UPR fitting", # 14
                  16384: "Pixel affected by ghost/crosstalk", # 15
}
# turn this into a dataframe that can be accessed with a boolean index
dq_flags_df = pd.DataFrame([(k,v) for k, v in dq_flags_descr.items()],
                           columns=['flag','descr'])
#dq_flags_df.sort_values(by='flag', ascending=False, inplace=True)
dq_flags_df.reset_index(drop=True, inplace=True)
"""
Pull out just the different flag values. Bonus points for listing the keys from highest to lowest, since it makes it easier to match the dq image values to the flags
"""
dq_flags = sorted(dq_flags_descr.keys())[::-1]

def dq_flag_parser_old(flag_int):
    """
    Convert a DQ HDU 32-bit word from an integer to base-2 and return the state of all the flags

    Parameters
    ----------
    flag_int : int
      the flag interpreted as an integer

    Returns
    -------
    flag_dict : dict
      a dict with True or False for each flag described in dq_flags
    """
    # flags have 16 bits
    flag_bin = f"{flag_int:0>16b}".replace(' ', '0')
    template_str = "1111 1111 1111 1111"
    template_str = template_str.replace(" ", "")
    # initialize the results dictionary
    flag_results = dict([(k, None) for k in dq_flags])
    if flag_bin[-1] == '0':
        flag_results[0] = True
    else:
        flag_results[0] = False
    # loop over the rest and use the index to match the key
    for i, val in enumerate(flag_bin[1:][::-1]):
        flag_key = 2**np.int(i)
        flag_results[flag_key] = (val == template_str[i])
    return flag_results

def dq_flag_parser(flag_int):
    """
    Convert a DQ HDU 32-bit word from an integer to base-2 and return the state of all the flags

    Parameters
    ----------
    flag_int : int
      the flag interpreted as an integer

    Returns
    -------
    flag_dict : dict
      a dict with True or False for each flag described in dq_flags
    """
    # flags have 16 bits
    flag_bin = f"{flag_int:0>16b}".replace(' ', '0')
    which_flags = list(map(lambda x: x=='1', flag_bin))
    if flag_bin[-1] == '0':
        which_flags[0] = True
    return dict(zip(dq_flags_df['flag'], which_flags)) 

"""
    # initialize the results dictionary
    flag_results = dict([(k, None) for k in dq_flags])
    if flag_bin[-1] == '0':
        flag_results[0] = True
    else:
        flag_results[0] = False
    # loop over the rest and use the index to match the key
    for i, val in enumerate(flag_bin[1:][::-1]):
        flag_key = 2**np.int(i)
        flag_results[flag_key] = (val == template_str[i])
    return flag_results
"""

