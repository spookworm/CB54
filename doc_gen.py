import os
from lib import portfolio_doc

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
