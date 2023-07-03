from fn_graph import Composer
import importlib
import graphviz
import os
from inspect import getmembers, isfunction


def composer_render(composer_call, path_doc, filename):
    import os
    composer_call.graphviz().render(directory=path_doc, filename=filename, format='png')
    os.remove(path_doc + filename)
    return composer_call


def composer_call(subfolder_name, module_name):
    module = importlib.import_module(f"{subfolder_name}.{module_name}")
    attributes = [f"{module_name}.{attr}" for attr in dir(module)]
    composer = Composer().update(attr for attr in attributes)
    # composer = Composer().update(ForwardBiCGSTABFFT.Aw)
    return composer

# import importlib
# subfolder_name = "lib"
# module_name = "ForwardBiCGSTABFFT"
# module = importlib.import_module(f"{subfolder_name}.{module_name}")
# attributes = [attr for attr in dir(module)]
# functions_list = {attr: getattr(module, attr) for attr in attributes if callable(getattr(module, attr))}


def workflow_doc(path_doc, filename):
    dot = graphviz.Digraph(
        node_attr={'shape': 'rectangle', 'color': '#FAB400', 'fillcolor': '#003A69', 'fontcolor': '#FAB400', 'style': 'filled'},
        edge_attr={'color': '#FAB400'}
        )
    dot.attr(label=r'\n\nDocumentation Workflow', labelloc="t", bgcolor='#003A69', fontcolor='#FAB400', fontsize='40', rankdir='LR')

    dot.node('./doc/0_MEng_Project_Portfolio_Cover_Pages_2023.pdf')
    dot.node('AnthonyJamesMcElwee_20211330_IEEE_Paper.pdf')
    dot.node('AnthonyJamesMcElwee_20211330_LR_Updated.pdf')
    dot.node('AnthonyJamesMcElwee_20211330_PDP_signed.pdf')
    dot.node('AnthonyJamesMcElwee_20211330_PRL.pdf')
    dot.node('AnthonyJamesMcElwee_20211330_RA.pdf')
    dot.node('AnthonyJamesMcElwee_20211330_PDI.pdf')
    dot.node('AnthonyJamesMcElwee_20211330_TR.pdf')
    dot.node('AnthonyJamesMcElwee_20211330_SCL.pdf')
    dot.node('AnthonyJamesMcElwee_20211330_FP.pdf')

    dot.edge('./doc/0_MEng_Project_Portfolio_Cover_Pages_2023.pdf', 'AnthonyJamesMcElwee_20211330_FP.pdf')
    dot.edge('AnthonyJamesMcElwee_20211330_IEEE_Paper.pdf', 'AnthonyJamesMcElwee_20211330_FP.pdf')
    dot.edge('AnthonyJamesMcElwee_20211330_LR_Updated.pdf', 'AnthonyJamesMcElwee_20211330_FP.pdf')
    dot.edge('AnthonyJamesMcElwee_20211330_PDP_signed.pdf', 'AnthonyJamesMcElwee_20211330_FP.pdf')
    dot.edge('AnthonyJamesMcElwee_20211330_PRL.pdf', 'AnthonyJamesMcElwee_20211330_FP.pdf')
    dot.edge('AnthonyJamesMcElwee_20211330_RA.pdf', 'AnthonyJamesMcElwee_20211330_FP.pdf')
    dot.edge('AnthonyJamesMcElwee_20211330_PDI.pdf', 'AnthonyJamesMcElwee_20211330_FP.pdf')
    dot.edge('AnthonyJamesMcElwee_20211330_TR.pdf', 'AnthonyJamesMcElwee_20211330_FP.pdf')
    dot.edge('AnthonyJamesMcElwee_20211330_SCL.pdf', 'AnthonyJamesMcElwee_20211330_FP.pdf')

    dot.render(directory=path_doc, filename=filename).replace('\\', '/')
    dot.render(directory=path_doc, filename=filename, view=False, format='png')
    os.remove(path_doc + filename + '.pdf')
    os.remove(path_doc + filename)


def workflow_framework(path_doc, filename):
    dot = graphviz.Digraph(
        node_attr={'shape': 'rectangle', 'color': '#FAB400', 'fillcolor': '#003A69', 'fontcolor': '#FAB400', 'style': 'filled'},
        edge_attr={'color': '#FAB400'}
        )
    dot.attr(label=r'\n\Framework Workflow', labelloc="t", bgcolor='#003A69', fontcolor='#FAB400', fontsize='40', rankdir='TB')

    dot.node('A')
    dot.node('B')
    dot.edge('A', 'B')

    dot.render(directory=path_doc, filename=filename).replace('\\', '/')
    dot.render(directory=path_doc, filename=filename, view=False, format='png')
    os.remove(path_doc + filename + '.pdf')
    os.remove(path_doc + filename)
