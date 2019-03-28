from pathlib import Path


def find_checkpoint(cfg):
    cp_iter = cfg['train']['checkpoint_every']
    steps = cfg['train']['steps']
    n_cp, fname_cp = 0, None
    for n_iter in range(cp_iter, steps + cp_iter, cp_iter):
        fname = cfg['train']['checkpoint_name'].format(n_iter=n_iter//cp_iter)
        if Path(fname).exists():
            n_cp, fname_cp = n_iter, fname
    return n_cp, fname_cp
