# -*- coding: utf-8 -*-
#
# IMPORT LIBRARIES:START
#
import os
import time
from fn_graph import Composer
import gradio as gr
from lib import graphviz_doc
from pypdf import PdfReader
#
# IMPORT LIBRARIES:END
#

#
# APP GLOBAL VARIABLES: START
#
path_doc = "./doc/"
seed = 42
#
# APP GLOBAL VARIABLES: END
#


#
# CUSTOM FUNCTIONS:START
#
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
#
# CUSTOM FUNCTIONS:END
#


#
# IMAGE & DOCUMENTATION RENDER: START
#
composer = graphviz_doc.composer_render(composer_call(), path_doc, "digraph")
graphviz_doc.workflow(path_doc, "workflow")
docx_time = os.path.getmtime("./doc/project_management/AnthonyJamesMcElwee_20211330_PRS.docx")
pdf_time = os.path.getmtime("AnthonyJamesMcElwee_20211330_PRS.pdf")
# Only compile reports if the docx file is fresher than the pdf
if docx_time - pdf_time >= 0.0:
    os.system("cmd_04_Compile_Report.bat")
#
# IMAGE & DOCUMENTATION RENDER: END
#
#
# GRADIO APP: START
#
with gr.Blocks(title="SolverEMF", analytics_enabled=True) as demo:
    #
    # SOLVER: START
    #
    with gr.Tab(label="Dev"):
        gr.HTML("<h1>Dev goes here</h1>")
        gr.HTML("""<h1>Last refresh: """ + str(time.ctime(time.time())) + """</h1>""")
    with gr.Tab(label="Dev Diagraph"):
        with gr.Column(scale=2):
            diagraph_image = gr.Image(value=path_doc + "digraph.png", type='pil')
            diagraph_image.style(height=600)
    #
    # SOLVER: END
    #
    #
    # PROJECT DOCUMENTATION: START
    #
    with gr.Tab(label="Github README"):
        with gr.Column(scale=2):
            readme_markdown = gr.Markdown(open("./README.md", 'r').read())
    with gr.Tab(label="Final Portfolio Compilation Files"):
        with gr.Row():
            with gr.Row():
                with gr.Column(scale=2):
                    gr.HTML("""<h1>workflow</h1>""")
                    workflow_image = gr.Image(value=path_doc + "workflow.png", type='pil')
                    workflow_image.style(height=600, width=600)
                with gr.Column(scale=2):
                    number_of_pages = len(PdfReader('AnthonyJamesMcElwee_20211330_FR.pdf').pages)
                    gr.HTML("""<h2>AnthonyJamesMcElwee_20211330_FR: """ + str(number_of_pages) + """ pages</h2>""")
                    doc_date = time.ctime(docx_time)
                    gr.HTML("""<h2>AnthonyJamesMcElwee_20211330_FR: """ + str(doc_date) + """ Last Modified</h2>""")
                    gr.HTML("""<h1>log_meetings_dates</h1>""")
                    gr.HTML(open("./doc/project_management/log_meetings_dates.html", 'r').read(), label="log_meetings_dates")
    #
    # PROJECT DOCUMENTATION: END
    #
    # The launch just seems to freeze everything in gradio mode. Leaving here as it may point to an issue at the sharing/hosting stage.
    # if __name__ == "__main__":
    #     demo.launch(server_port=8080)
#
# GRADIO APP: END
#
