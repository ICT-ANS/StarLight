import os
import sys, time
from graphviz import Digraph
from collections import namedtuple
from nas.classification_darts.classify_utils import convert_pdf_to_img
from nas.classification_darts.classify_utils import from_darts_read_log
from PyQt5.QtCore import QThread, pyqtSignal
from nas.classification_darts.nas_classify_darts_quiver.quiver_engine.model_utils import register_hook
from nas.classification_darts.nas_classify_darts_quiver.quiver_engine import server
import threading
from nas.classification_darts.nas_classify_darts_quiver.searched_models.model_generator import model_builder
from config import C


def plot(genotype, filename):
    """
    Plot the searched cell to the designated name.

    Parameters
    ----------
    genotype : Genotype
        The searched cell genotype needs to be plotted
    filename : str
        The file name of saving the cell figure

    Returns
    -------
    None
    """
    g = Digraph(
        format='pdf',
        edge_attr=dict(fontsize='20', fontname="times"),
        node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5',
                       penwidth='2', fontname="times"),
        engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
        g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
        for k in [2 * i, 2 * i + 1]:
            op, j = genotype[k]
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j - 2)
            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    g.render(filename, view=False, cleanup=True)


def plot_and_convert_genotype(genotype, name, output):
    """
    Plot the searched cell.

    Parameters
    ----------
    genotype :
        The searched cell genotype needs to plot
    name : string
        The name of which to plot
    output : string
        The path to output

    Returns
    -------
    None
    """
    if type(genotype) is str:
        from collections import namedtuple
        Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
        genotype = eval(genotype)
    plot(genotype.normal, "%s/%s_normal_cell" % (output, name))
    plot(genotype.reduce, "%s/%s_reduction_cell" % (output, name))
    convert_pdf_to_img(inputDir=output, outputDir=output)


def generate_all_genotypes(root_file_path, method_list, method_index_list, dataset='cifar10', seed=2):
    """
    This function generates all the searched genotype figures according to the provided method list and their log path.

    Parameters
    ----------
    root_file_path : str
        the direction which saves all the log
    method_list : list
        the names of different search algorithms
    method_index_list : list
        the corresponding relationship of each method
    dataset : str
        the corresponding dataset name
    seed : int
        seed of random number

    Returns
    -------
    None
    """
    data_dict = {}
    for _m in method_list:
        data_dict[_m] = from_darts_read_log('{}/logdir/{}_{}_{}.log'.format(root_file_path, _m, dataset, seed),
                                            key_words=method_index_list)
        if not os.path.exists('{}/figure'.format(root_file_path)):
            os.mkdir('{}/figure'.format(root_file_path))
        if not os.path.exists('{}/figure/{}'.format(root_file_path, _m)):
            os.mkdir('{}/figure/{}'.format(root_file_path, _m))
        for i, genotype in enumerate(data_dict[_m]['genotype']):
            print('method: {}, genotype: {}'.format(_m, i))
            plot_and_convert_genotype(genotype, name=str(i), output='{}/figure/{}'.format(root_file_path, _m))


class modelVisTask(QThread):
    signalUpdateUi = pyqtSignal()  # 定义更新UI信号

    def __init__(self, mainW, dataset, model):
        """
        This class defines the model visualization task.

        Parameters
        ----------
        mainW : UI
            the UI to show the model visualization
        dataset : str
            dataset name
        model : nn.Module
            the model to be shown
        """
        super().__init__()
        self.cur_dataset = dataset
        self.signalUpdateUi.connect(mainW.modelVisLoad)
        self.sign = True

        self.model, self.input_size = model_builder(model)

        self.hook_list = register_hook(self.model)

    def stop(self):
        """
        This function stop the visualization task.

        Returns
        -------
        None
        """
        self._stop()
        # thread = threading.Thread(target=self._stop)
        # thread.start()

    def _stop(self):
        """
        This function stop the visualization task.

        Returns
        -------
        None
        """
        self.http_server.stop()
        count = 0
        while True and count < 600:
            time.sleep(0.1)
            if self.http_server.closed:
                break
            count += 1
            print('stop: {}'.format(count))
        self.exit()

    def run(self):
        """
        Execute function.

        Returns
        -------
        None
        """
        thread = threading.Thread(target=self.load)
        thread.start()
        self.http_server = server.get_server(
            self.model,
            self.hook_list,
            os.path.join(C.quiver_data, self.cur_dataset),
            port=5050
        )
        self.http_server.serve_forever()

    def load(self):
        """
        This function loads the visualization html page.

        Returns
        -------
        None
        """
        # self.signalUpdateUi.emit()
        count = 0
        while True and count < 6000:
            time.sleep(0.1)
            if hasattr(self, 'http_server') and self.http_server.started:
                self.signalUpdateUi.emit()
                if self.http_server.real_started:
                    break
            count += 1
            print('load: {}'.format(count))


if __name__ == '__main__':
    Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

    # genotype = eval( "Genotype(normal=[('dil_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 2),
    # ('sep_conv_3x3', 0), " "('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4), ('sep_conv_5x5', 3)],
    # normal_concat=range(2, " "6), reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('skip_connect', 2),
    # ('max_pool_3x3', 0), " "('max_pool_3x3', 0), ('dil_conv_3x3', 3), ('skip_connect', 2), ('sep_conv_5x5', 4)],
    # reduce_concat=range(2, " "6)) ") genotype = "Genotype(normal=[('dil_conv_5x5', 1), ('max_pool_3x3', 0),
    # ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4),
    # ('sep_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1),
    # ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('dil_conv_3x3', 3), ('skip_connect', 2),
    # ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))"

    # plot(genotype.normal, "figure/normal_cell")
    # plot(genotype.reduce, "figure/reduction_cell")
    #
    # convert_pdf_to_img(inputDir='./figure', outputDir='./figure')

    method_list = ['darts', 'pdarts', 'pcdarts', 'sdarts_rs', 'sdarts_pgd', 'sgas_cri1', 'sgas_cri2']
    method_index_list = ['train_loss', 'train_acc', 'valid_loss', 'valid_acc', 'genotype']
    generate_all_genotypes(method_list, method_index_list, dataset='cifar10', seed=2)


