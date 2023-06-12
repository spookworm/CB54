# -*- coding: utf-8 -*-
#
# IMPORT LIBRARIES:START
#
import os
import time
import gradio as gr
from pypdf import PdfReader
from lib import graphviz_doc
from lib import portfolio_doc
from lib import geo_gen
#
# IMPORT LIBRARIES:END
#

#
# APP GLOBAL VARIABLES: START
#
seed = 42
path_doc = "./doc/"
path_geo = "./code_ref/vefie_for_building_streamlined/geometry/"
path_lut = "./code_ref/vefie_for_building_streamlined/lut/materials.json"
object_name = 'object_mp_landscape_empty.txt'
#
# APP GLOBAL VARIABLES: END
#
#
# CUSTOM FUNCTIONS:START
#


def composer_call():
    from fn_graph import Composer
    composer_1 = (
        Composer()
        .update(
            # list of custom functions goes here
            geo_gen.epsilon0,
            geo_gen.mu0,
            geo_gen.realmax,
            geo_gen.input_carrier_frequency,
            geo_gen.input_disc_per_lambda,
            geo_gen.angular_frequency,
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
graphviz_doc.workflow_framework(path_doc, "workflow_framework")
graphviz_doc.workflow_doc(path_doc, "workflow_doc")
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
last_refresh = time.ctime(time.time())
with gr.Blocks(title="SolverEMF", analytics_enabled=True) as demo:
    #
    # SOLVER: START
    #
    with gr.Tab(label="Dev"):
        gr.HTML("""<h1>Last refresh: """ + str(last_refresh) + """</h1>""")
        gr.HTML("<h1>Dev Diagraph</h1>")
        diagraph_image = gr.Image(value=path_doc + "digraph.png", type='pil')
        diagraph_image.style(height=600)

        gr.Number(value=composer.epsilon0(), label="composer.epsilon0()")
        gr.Number(value=composer.mu0(), label="composer.mu0()")
        gr.Number(value=composer.realmax(), label="composer.realmax()")
        gr.Number(value=composer.input_carrier_frequency(), label="composer.input_carrier_frequency()")
        gr.Textbox(value=composer.input_disc_per_lambda, label="composer.input_disc_per_lambda")
        gr.Number(value=composer.angular_frequency(), label="composer.angular_frequency()")
    #
    # SOLVER: END
    #
    #
    # PROJECT DOCUMENTATION: START
    #
    with gr.Tab(label="Github README"):
        gr.HTML("""<h1>Last refresh: """ + str(last_refresh) + """</h1>""")
        with gr.Column(scale=2):
            readme_markdown = gr.Markdown(open("./README.md", 'r').read())
            number_of_pages = len(PdfReader('AnthonyJamesMcElwee_20211330_FP.pdf').pages)
            gr.HTML("""<h2>AnthonyJamesMcElwee_20211330_FP Pages: """ + str(number_of_pages) + """</h2>""")
            doc_date = time.ctime(os.path.getmtime("AnthonyJamesMcElwee_20211330_FP.pdf"))
            gr.HTML("""<h2>AnthonyJamesMcElwee_20211330_FP Last Modified: """ + str(doc_date) + """</h2>""")
    with gr.Tab(label="Things missing from final report"):
        gr.HTML("""<h1>Last refresh: """ + str(last_refresh) + """</h1>""")
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML("""<h1>mathjax derivations</h1>""")
                gr.Markdown(open("./doc/mathjax_int.md", 'r').read())
                gr.HTML("""<h1>workflow_framework</h1>""")
                workflow_image = gr.Image(value=path_doc + "workflow_framework.png", type='pil')
                gr.HTML("""<h1>workflow_doc</h1>""")
                workflow_image = gr.Image(value=path_doc + "workflow_doc.png", type='pil')
                workflow_image.style(height=600, width=600)
                gr.HTML("""<h1>log_meetings_dates</h1>""")
                gr.HTML(open("./doc/log_meetings_dates.html", 'r').read(), label="log_meetings_dates")
                diagraph_image = gr.Image(value=path_doc + "digraph.png", type='pil')
                diagraph_image.style(height=600)
    #
    # PROJECT DOCUMENTATION: END
    #
    # The launch just seems to freeze everything in gradio mode. Leaving here as it may point to an issue at the sharing/hosting stage.
    # if __name__ == "__main__":
    #     demo.launch(server_port=8080)
#
# GRADIO APP: END
#
