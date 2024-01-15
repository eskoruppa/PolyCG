import numpy as np
import sys
from typing import List, Tuple, Callable, Any, Dict
import matplotlib.pyplot as plt

from .SO3.so3 import euler2rotmat, rotmat2euler

# from .conversions import conf2vecs, conf2rotmats, triads2rotmats, rotmats2triads
from .conversions import *
from .bps_params import seq2rotbps


# To do: implement Configs class that automatically handles proper application of rotations
# class Configs


def bpsrot_fluct2config(
    Thetas_fluct: np.ndarray,
    Thetas_gs: np.ndarray,
    include_triads=True,
    include_positions=True,
    rotation_map="euler",
    split_fluctuations="vector",
    disc_len=0.34,
) -> Dict[str, Any]:
    """Generate configurations from ensemble of fluctuation states

    Args:
        Thetas_fluct (np.ndarray): Fluctuating components of rotational bps parameters (...,N,3)
        Thetas_gs (np.ndarray): Static components of rotational bps parameters (N,3)
        include_triads (bool): (default: True)
        include_positions (bool): (default: True)
        rotation_map (str): (default: 'euler')
        split_fluctuations (str): (default: 'vector')
        disc_len (float): (default: 0.34)

    Returns:
        Dict[Any]: Configurations
    """

    if (
        split_fluctuations.lower()
        in ["rotmat", "mat", "matrix", "rotationmatrix", "group", "liegroup"]
        or split_fluctuations == "SO3"
    ):
        if rotation_map.lower() != "euler":
            raise ValueError(
                f"Option for matrix decomposition of static and dynamic components is only available for Euler rotation map."
            )
        vecsplit = False
    elif (
        split_fluctuations.lower() in ["vec", "vector", "algebra", "liealgebra"]
        or split_fluctuations == "so3"
    ):
        vecsplit = True
    else:
        raise ValueError(
            f'Unknown value "{split_fluctuations}" for split_fluctuations.'
        )

    # Cayley map option
    if rotation_map.lower() == "cayley":
        Thetas = Thetas_fluct + Thetas_gs
        rotmats = cayleys2rotmats(Thetas)

    # Euler map option
    elif rotation_map.lower() == "euler":
        if vecsplit:
            Thetas = Thetas_fluct + Thetas_gs
            rotmats = eulers2rotmats(Thetas)
        else:
            rotmats = eulers2rotmats_SO3fluct(Thetas_gs, Thetas_fluct)
    else:
        raise ValueError(f'Unknown rotation_map "{rotation_map}"')

    triads = rotmats2triads(rotmats)
    positions = triads2positions(triads)

    configs = dict()
    configs["Thetas_fluct"] = Thetas_fluct
    configs["Thetas_gs"] = Thetas_gs
    if include_triads:
        configs["triads"] = triads
    if include_positions:
        configs["positions"] = positions
    return configs


def plot_config(configs: Dict[str, Any], confid=0):
    conf = configs["positions"][confid] * 10
    triads = configs["triads"][confid]
    print(conf.shape)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    # ax.set_box_aspect((np.ptp(conf), np.ptp(conf), np.ptp(conf)))  # aspect ratio is 1:1:1 in data space

    ax.scatter(
        conf[:, 0],
        conf[:, 1],
        conf[:, 2],
        zdir="z",
        s=20,
        alpha=0.5,
        c="black",
        depthshade=True,
    )

    e1 = triads[:, :, 0]
    e2 = triads[:, :, 1]
    e3 = triads[:, :, 2]

    ax.quiver(
        conf[:, 0], conf[:, 1], conf[:, 2], e1[:, 0], e1[:, 1], e1[:, 2], color="blue"
    )  # , color=['r','b','g'], scale=21)
    ax.quiver(
        conf[:, 0], conf[:, 1], conf[:, 2], e2[:, 0], e2[:, 1], e2[:, 2], color="green"
    )  # , color=['r','b','g'], scale=21)
    ax.quiver(
        conf[:, 0], conf[:, 1], conf[:, 2], e3[:, 0], e3[:, 1], e3[:, 2], color="red"
    )  # , color=['r','b','g'], scale=21)

    # ax.plot(conf[:, 0], conf[:, 1], conf[:, 2], zdir='z', c='black',lw=3)

    min = np.min(conf)
    max = np.max(conf)
    rge = max - min
    overhead = 0.05
    for i in range(3):
        mid = np.mean(conf[:, i])
        lo = mid - (0.5 + overhead) * rge
        hi = mid + (0.5 + overhead) * rge
        if i == 0:
            ax.set_xlim([lo, hi])
        if i == 1:
            ax.set_ylim([lo, hi])
        if i == 2:
            ax.set_zlim([lo, hi])
    plt.show()
