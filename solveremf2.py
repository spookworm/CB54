# -*- coding: utf-8 -*-
# IMPORT LIBRARIES:START
from fn_graph import Composer
import gradio as gr
from lib import graphviz_doc
# IMPORT LIBRARIES:END

# APP GLOBAL VARIABLES: START
path_doc = "./doc/"
seed = 42
# APP GLOBAL VARIABLES: END


# CUSTOM FUNCTIONS:START
# PUT unput_ AT THE START OF FUNCTIONS THAT REQUIRE VARIABLE INPUTS
def composer_call():
    composer_1 = (
        Composer()
        .update(
            # list of custom functions goes here
        )
        # .update_parameters(input_length_side=input_length_x_side)
        # .cache()
    )
    return composer_1
# CUSTOM FUNCTIONS:END


# IMAGE RENDER: START
composer = graphviz_doc.composer_render(composer_call(), path_doc, "digraph")
graphviz_doc.workflow(path_doc, "workflow")
# IMAGE RENDER: END

# GRADIO APP: START
with gr.Blocks(title="SolverEMF", analytics_enabled=True) as demo:
    with gr.Tab(label="Github README"):
        with gr.Column(scale=2):
            readme_markdown = gr.Markdown(open("./README.md", 'r').read())
    with gr.Tab(label="Dev"):
        gr.HTML("<h1>Dev goes here</h1>")
    with gr.Tab(label="Dev Diagraph"):
        with gr.Column(scale=2):
            diagraph_image = gr.Image(value=path_doc + "digraph.png", type='pil')
            diagraph_image.style(height=600)
    with gr.Tab(label="Final Portfolio Compilation Files"):
        with gr.Row():
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML("""<h1>workflow</h1>""")
                    workflow_image = gr.Image(value=path_doc + "workflow.png", type='pil')
                    workflow_image.style(height=600, width=600)
                with gr.Column(scale=2):
                    gr.HTML("""<h1>log_meetings_dates</h1>""")
                    gr.HTML(open("./doc/project_management/log_meetings_dates.html", 'r').read(), label="log_meetings_dates")
# GRADIO APP: END
