def net_factory(net_type="unet", in_chns=3, class_num=4, args=None):

    if net_type == "unet":
        from networks.unet import UNet
        net = UNet(in_chns=in_chns, class_num=class_num)
    elif net_type == "resunet":
        from networks.resunet import Resunet
        net = Resunet(in_chns=in_chns, class_num=class_num)
    else:
        print("error model")
        exit()
    return net