from yacs.config import CfgNode as CN


_C = CN()

_C.DATA = CN()
_C.DATA.DATA_LOC = "" #operating system specific 

_C.PARAMETERS = CN()
_C.PARAMETERS.EPOCHS = 5



def get_cfg_defaults():
    return _C.clone()