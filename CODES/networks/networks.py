import torch.nn as nn
import torch


class NetworksFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(network_name, *args, **kwargs):
        ################ Ours #################
        if network_name == 'Ours_Reconstruction':
            from networks.Ours_Reconstruction import Net
            network = Net(*args, **kwargs)
        elif network_name == 'Ours_DeblurOnly':
            from networks.Ours_DeblurOnly import Net
            network = Net(*args, **kwargs)
        else:
            raise ValueError("Network %s not recognized." % network_name)

        # print(network)
        print("Network %s was created: " % network_name)
        print('Network parameters: {}'.format(sum([p.data.nelement() for p in network.network.parameters()])))

        return network


class NetworkBase(nn.Module):
    def __init__(self):
        super(NetworkBase, self).__init__()
        self._name = 'BaseNetwork'

    @property
    def name(self):
        return self._name
