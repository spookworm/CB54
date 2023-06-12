def workflow(path_doc, filename):
    import graphviz
    import os

    dot = graphviz.Digraph(
        node_attr={'shape': 'rectangle', 'color': '#FAB400', 'fillcolor': '#003A69', 'fontcolor': '#FAB400', 'style': 'filled'},
        edge_attr={'color': '#FAB400'}
        )
    dot.attr(label=r'\n\nDocumentation Workflow', labelloc="t", bgcolor='#003A69', fontcolor='#FAB400', fontsize='40')

    dot.node('Zotero References')
    dot.node('AnthonyJamesMcElwee_20211330_FR.docx')
    dot.node('AnthonyJamesMcElwee_20211330_PRS.docx')
    dot.edge('AnthonyJamesMcElwee_20211330_PRS.docx', 'AnthonyJamesMcElwee_20211330_FR.docx')
    dot.edge('Zotero References', 'AnthonyJamesMcElwee_20211330_FR.docx')
    dot.edge('AnthonyJamesMcElwee_20211330_FR.docx', 'AnthonyJamesMcElwee_20211330_FR.pdf')

    dot.render(directory=path_doc, filename=filename).replace('\\', '/')
    dot.render(directory=path_doc, filename=filename, view=False, format='png')
    os.remove(path_doc + filename + '.pdf')
    os.remove(path_doc + filename)


def composer_render(composer_call, path_doc, filename):
    import os
    composer_call.graphviz().render(directory=path_doc, filename=filename, format='png')
    os.remove(path_doc + filename)
    return composer_call
