import numpy as np
import sys
from typing import List, Tuple, Callable, Any, Dict
from .SO3.so3 import euler2rotmat, rotmat2euler

# from .conversions import conf2vecs, conf2rotmats, triads2rotmats, rotmats2triads
from .conversions import *
from .bps_params import seq2rotbps
from .configs import bpsrot_fluct2config


def bpsrot_sample(
    seq: str,
    num_confs: int,
    include_triads=True,
    include_positions=True,
    rotation_map="euler",
    split_fluctuations="vector",
    disc_len=0.34,
    ps_set="default",
) -> Dict[str, Any]:
    gs, stiff = seq2rotbps(
        seq,
        in_nm=True,
        disc_len=disc_len,
        rotation_map=rotation_map,
        split_fluctuations=split_fluctuations,
        gs_units="rad",
        ps_set=ps_set,
    )

    covmat = np.linalg.inv(stiff / disc_len)
    dThetas = sample_free(covmat, num_confs, groundstate=None)

    gs_vecs = statevec2vecs(gs)
    dThetas_vecs = statevec2vecs(dThetas)

    configs = bpsrot_fluct2config(
        dThetas_vecs,
        gs_vecs,
        include_triads=include_triads,
        include_positions=include_positions,
        rotation_map=rotation_map,
        split_fluctuations=split_fluctuations,
        disc_len=disc_len,
    )
    return configs


def sample_free(covmat: np.ndarray, num_confs: int, groundstate=None) -> np.ndarray:
    dx = np.random.multivariate_normal(np.zeros(len(covmat)), covmat, num_confs)
    if groundstate is not None:
        dx += groundstate
    return dx


def sample_cg(
    stiffmat: np.ndarray,
    groundstate: np.ndarray,
    cgsteps: int,
    num_confs: int,
    omit_mismatching_tail=False,
    disc_len=0.34,
) -> np.ndarray:
    if not omit_mismatching_tail and (len(stiffmat) // 3) % cgsteps != 0:
        raise ValueError(
            f"Number of bp steps needs to be a multiple of the coarse-graining step (cgstep). To force coarse graining by discarding the mismatching tail set omit_mismatching_tail to True."
        )

    # print(len(stiffmat)//3)

    covmat = np.linalg.inv(stiffmat / disc_len)
    statevecs = sample_free(covmat, num_confs, groundstate=groundstate)

    vecs = statevec2vecs(statevecs)
    test_statevecs = vecs2statevec(vecs)

    print(np.sum(statevecs - test_statevecs))

    rotmats = cayleys2rotmats(vecs)

    test_vecs = rotmats2cayleys(rotmats)
    print(np.sum(vecs - test_vecs))

    triads = rotmats2triads(rotmats)
    test_rotmats = triads2rotmats(triads)

    print(np.sum(rotmats - test_rotmats))

    eulers = cayleys2eulers(vecs)

    cayleys = eulers2cayleys(eulers)

    print(np.sum(cayleys - vecs))

    Rc2e = cayleys2eulers_lintrans(vecs2statevec(cayleys))
    Re2c = eulers2cayleys_lintrans(vecs2statevec(eulers))

    print(np.sum(np.eye(len(Rc2e)) - np.matmul(Rc2e, Re2c)))

    Rc2e = cayleys2eulers_lintrans(groundstate)

    gs_euler = vecs2statevec(cayleys2eulers(statevec2vecs(groundstate)))
    gs_euler2 = np.matmul(Rc2e, groundstate)

    print(np.sum(gs_euler - gs_euler2))

    # rotmats = conf2rotmats(conf)

    # for rotmat in rotmats:
    #     print(rotmat)

    # print(conf.shape)

    # rotmats = conf2rotmat(conf)

    # print(rotmats.shape)

    # triads = rotmats2triads(rotmats)

    # print(triads[-1,-1])

    # print(triads.shape)
    # check_rotmats = triads2rotmats(triads)
    # print(check_rotmats.shape)
    # print(rotmats.shape)

    # print('###########')
    # print('rotmats')
    # for i in range(len(rotmats)):
    #     print('----------------')
    #     for j in range(len(rotmats[0])):
    #         print(np.sum(rotmats[i,j]-check_rotmats[i,j]))

    #     print(rotmats[-1,-1])
    #     print(check_rotmats[-1,-1])

    # print(np.sum(rotmats-check_rotmats))

    # sys.exit()

    # check_triads = rotmats2triads(check_rotmats)

    # print(check_triads.shape)
    # print(check_rotmats.shape)

    # print('###########')
    # print('triads')
    # for i in range(len(triads)):
    #     print('----------------')
    #     for j in range(len(triads[0])):
    #         print(triads[i,j]-check_triads[i,j])

    # # print(np.sum(check_rotmats-rotmats))

    # # [:,::cgsteps]
