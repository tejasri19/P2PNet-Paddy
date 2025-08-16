#from .p2pnet_resnet import build
#from .p2pnet import build
#from .p2pnet_mobilenetv3 import build
#from .rsp2pnet import build
from .p2pnet_efficientnetlite import build

def build_model(args, training=False):
    return build(args, training)