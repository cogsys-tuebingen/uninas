"""
export the network via onnx
"""

from uninas.models.networks.uninas.abstract import AbstractUninasNetwork
from uninas.utils.shape import Shape
from uninas.utils.loggers.python import LoggerManager, log_headline
from uninas.utils.torch.standalone import get_network
from uninas.utils.paths import replace_standard_paths
from uninas.register import Register


def example_export_network(path: str) -> AbstractUninasNetwork:
    """ create a new network and export it, does not require to have onnx installed """
    network = get_network("FairNasC", Shape([3, 224, 224]), Shape([1000]), weights_path=None)
    network = network.cuda()
    network.export_onnx(path, export_params=True)
    return network


try:
    import onnx

    if __name__ == '__main__':
        logger = LoggerManager().get_logger()
        export_path = replace_standard_paths("{path_tmp}/onnx/FairNasC.onnx")
        net1 = example_export_network(export_path)

        log_headline(logger, "onnx graph")
        net2 = onnx.load(export_path)
        onnx.checker.check_model(net2)
        logger.info(onnx.helper.printable_graph(net2.graph))


except ImportError as e:
    Register.missing_import(e)
