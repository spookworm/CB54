# -*- coding: utf-8 -*-
#
# IMPORT LIBRARIES:START
#
import os
import time
from fn_graph import Composer
import gradio as gr
from pypdf import PdfReader
import pypdftk
from lib import graphviz_doc
from lib import portfolio_doc
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
# Portfolio report update section
# Converter found at https://github.com/cognidox/OfficeToPDF/releases
portfolio_doc.docx_compile(".\\doc\\0_MEng_Project_Portfolio_Cover_Pages_2023.docx", ".\\doc\\0_MEng_Project_Portfolio_Cover_Pages_2023.pdf")
portfolio_doc.docx_compile(".\\doc\\IEEE_Paper\\0_MEng_Project_Paper_Cover_Pages_2023.docx", ".\\doc\\IEEE_Paper\\0_MEng_Project_Paper_Cover_Pages_2023.pdf")
portfolio_doc.docx_compile(".\\doc\\IEEE_Paper\\1_IEEE_Template.docx", ".\\doc\\IEEE_Paper\\1_IEEE_Template.pdf")
portfolio_doc.portfolio_compile([".\\doc\\IEEE_Paper\\0_MEng_Project_Paper_Cover_Pages_2023.pdf", ".\\doc\\IEEE_Paper\\1_IEEE_Template.pdf"], "AnthonyJamesMcElwee_20211330_IEEE_Paper.pdf")
portfolio_doc.docx_compile(".\\doc\\Literature_Review\\0_MEng_Project_Literature_Review_Cover_Pages_2023.docx", ".\\doc\\Literature_Review\\0_MEng_Project_Literature_Review_Cover_Pages_2023.pdf")
portfolio_doc.docx_compile(".\\doc\\Literature_Review\\1_IEEE_Template.docx", ".\\doc\\Literature_Review\\1_IEEE_Template.pdf")
portfolio_doc.portfolio_compile([".\\doc\\Literature_Review\\0_ MEng_Project_Literature_Review_Cover_Pages_2023.pdf", ".\\doc\\Literature_Review\\1_IEEE_Template.pdf"], "AnthonyJamesMcElwee_20211330_LR_Updated.pdf")
portfolio_doc.docx_compile(".\\doc\\Project_Design_Plan\\AnthonyJamesMcElwee_20211330_PDP_signed.docx", "AnthonyJamesMcElwee_20211330_PDP_signed.pdf")
portfolio_doc.docx_compile(".\\doc\\Project_Research_Log\\AnthonyJamesMcElwee_20211330_PRL.docx", "AnthonyJamesMcElwee_20211330_PRL.pdf")
portfolio_doc.docx_compile(".\\doc\\Risk_Assignment\\AnthonyJamesMcElwee_20211330_RA.docx", "AnthonyJamesMcElwee_20211330_RA.pdf")
os.system("echo I think that these remaining files will have to be created in the main folder to include subfolder material. This may aksi be the case for the IEEE_Paper too.")
os.system("echo Missing AnthonyJamesMcElwee_20211330_PDI.pdf")
os.system("echo Missing AnthonyJamesMcElwee_20211330_TR.pdf")
os.system("echo Missing AnthonyJamesMcElwee_20211330_SCL needs to include all relevant github code and GITHUB README.md file also.")
os.system("echo Missing AnthonyJamesMcElwee_20211330_SCL.pdf")
portfolio_doc.portfolio_compile(["./doc/0_MEng_Project_Portfolio_Cover_Pages_2023.pdf", "AnthonyJamesMcElwee_20211330_IEEE_Paper.pdf", "AnthonyJamesMcElwee_20211330_LR_Updated.pdf", "AnthonyJamesMcElwee_20211330_PDP_signed.pdf", "AnthonyJamesMcElwee_20211330_PRL.pdf", "AnthonyJamesMcElwee_20211330_RA.pdf"], "AnthonyJamesMcElwee_20211330_FP.pdf")
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
                # with gr.Column(scale=2):
                #     number_of_pages = len(PdfReader('AnthonyJamesMcElwee_20211330_FP.pdf').pages)
                #     gr.HTML("""<h2>AnthonyJamesMcElwee_20211330_FP: """ + str(number_of_pages) + """ pages</h2>""")
                #     doc_date = time.ctime(os.path.getmtime("AnthonyJamesMcElwee_20211330_FP.pdf"))
                #     gr.HTML("""<h2>AnthonyJamesMcElwee_20211330_FP: """ + str(doc_date) + """ Last Modified</h2>""")
                #     gr.HTML("""<h1>log_meetings_dates</h1>""")
                #     gr.HTML(open("./doc/log_meetings_dates.html", 'r').read(), label="log_meetings_dates")
    #
    # PROJECT DOCUMENTATION: END
    #
    # The launch just seems to freeze everything in gradio mode. Leaving here as it may point to an issue at the sharing/hosting stage.
    # if __name__ == "__main__":
    #     demo.launch(server_port=8080)
#
# GRADIO APP: END
#
