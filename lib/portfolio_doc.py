def docx_compile(docA, pdfA):
    import os
    if os.path.exists(pdfA) is False:
        os.system(".\\env\\officetopdf.exe " + str(docA) + " " + str(pdfA))
    else:
        docA_time = os.path.getmtime(docA)
        pdfA_time = os.path.getmtime(pdfA)
        if docA_time - pdfA_time >= 0.0:
            os.system(".\\env\\officetopdf.exe " + str(docA) + " " + str(pdfA))


def portfolio_compile(pdf_list_input, pdf_list_output):
    # only concat if output is older than input
    import os
    import pypdftk
    if os.path.exists(pdf_list_output) is False:
        pypdftk.concat(pdf_list_input, pdf_list_output)
    else:
        seconds_list = list()
        for pdf in pdf_list_input:
            seconds_list.append(os.path.getmtime(pdf))
        if max(seconds_list) > os.path.getmtime(pdf_list_output):
            pypdftk.concat(pdf_list_input, pdf_list_output)

# OTHER CODE THAT MIGHT BE USEFUL
# FOR %%g IN (*.pdf) DO (
#    (Echo "%%g" | FIND /I "2_Optional_Bibliography_Master" 1>NUL) || (
# 		del "%%g"
#    )
# )

# pandoc README.md _o README.docx
# pandoc _f docx _t gfm AnthonyJamesMcElwee_20211330_FR.docx _o AnthonyJamesMcElwee_20211330_FR.md
# pandoc _f docx _t markdown_mmd AnthonyJamesMcElwee_20211330_FR.docx _o AnthonyJamesMcElwee_20211330_FR.md
# pandoc _f docx _t markdown AnthonyJamesMcElwee_20211330_FR.docx _o AnthonyJamesMcElwee_20211330_FR.md
# pandoc _f docx _t markdown_strict AnthonyJamesMcElwee_20211330_FR.docx _o AnthonyJamesMcElwee_20211330_FR.md
# pandoc _f docx _t markdown_phpextra AnthonyJamesMcElwee_20211330_FR.docx _o AnthonyJamesMcElwee_20211330_FR.md
# pandoc _f docx _t commonmark AnthonyJamesMcElwee_20211330_FR.docx _o AnthonyJamesMcElwee_20211330_FR.md

# pandoc __extract_media=. _s AnthonyJamesMcElwee_20211330_FR.docx _t html _c styles.css _o AnthonyJamesMcElwee_20211330_FR.html
# powershell _Command "(gc AnthonyJamesMcElwee_20211330_FR.html) _replace './media/', 'file/media/' | Out_File _encoding ASCII AnthonyJamesMcElwee_20211330_FR.html"
