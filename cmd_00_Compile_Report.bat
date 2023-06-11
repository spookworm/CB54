@echo off
call conda activate solveremf2

echo converter found at https://github.com/cognidox/OfficeToPDF/releases
echo DOCX to PDF will only run in Gradio if the DOCX is newer thant the PDF version. TO BE FULLY IMPLEMENTED

"./env/officetopdf.exe" "./doc/Project_Research_Log/AnthonyJamesMcElwee_20211330_PRL.docx" "AnthonyJamesMcElwee_20211330_PRL.pdf"

"./env/officetopdf.exe" "./doc/Project_Design_Plan/AnthonyJamesMcElwee-20211330-PDP-signed.docx" "AnthonyJamesMcElwee-20211330-PDP-signed.pdf"

"./env/officetopdf.exe" "./doc/Risk_Assignment/AnthonyJamesMcElwee_20211330_RA.docx" "AnthonyJamesMcElwee_20211330_RA.pdf"

"./env/officetopdf.exe" "./doc/Literature_Review/0_ MEng-Project-Literature-Review-Cover-Pages-2023.docx" "./doc/Literature_Review/0_ MEng-Project-Literature-Review-Cover-Pages-2023.pdf"
"./env/officetopdf.exe" "./doc/Literature_Review/1_IEEE_Template.docx" "./doc/Literature_Review/1_IEEE_Template.pdf"
pdftk "./doc/Literature_Review/0_ MEng-Project-Literature-Review-Cover-Pages-2023.pdf" "./doc/Literature_Review/1_IEEE_Template.pdf" cat output AnthonyJamesMcElwee_20211330_LR_Updated.pdf

"./env/officetopdf.exe" "./doc/IEEE_Paper/0_ MEng-Project-Paper-Cover-Pages-2023.docx" "./doc/IEEE_Paper/0_ MEng-Project-Paper-Cover-Pages-2023.pdf"
"./env/officetopdf.exe" "./doc/IEEE_Paper/1_IEEE_Template.docx" "./doc/IEEE_Paper/1_IEEE_Template.pdf"
pdftk "./doc/IEEE_Paper/0_ MEng-Project-Paper-Cover-Pages-2023.pdf" "./doc/IEEE_Paper/1_IEEE_Template.pdf" cat output AnthonyJamesMcElwee_20211330_IEEE_Paper.pdf

"./env/officetopdf.exe" "./doc/0_ MEng-Project-Portfolio-Cover-Pages-2023.docx" "./doc/0_ MEng-Project-Portfolio-Cover-Pages-2023.pdf"

pdftk ^
"./doc/0_ MEng-Project-Portfolio-Cover-Pages-2023.pdf" ^
AnthonyJamesMcElwee_20211330_IEEE_Paper.pdf ^
AnthonyJamesMcElwee_20211330_LR_Updated.pdf ^
AnthonyJamesMcElwee-20211330-PDP-signed.pdf ^
AnthonyJamesMcElwee_20211330_PRL.pdf ^
AnthonyJamesMcElwee_20211330_RA.pdf ^
cat output AnthonyJamesMcElwee_20211330_FP.pdf

REM REM I think that these remaining files will have to be created in the main folder to include subfolder material.
REM REM This may aksi be the case for the IEEE_Paper too.
REM AnthonyJamesMcElwee_20211330_PDI.pdf
REM AnthonyJamesMcElwee_20211330_TR.pdf
REM AnthonyJamesMcElwee_20211330_SCL needs to include all relevant github code and GITHUB README.md file also.
REM AnthonyJamesMcElwee_20211330_SCL.pdf
echo end of compilation
PAUSE

REM OTHER CODE THAT MIGHT BE USEFUL
REM FOR %%g IN (*.pdf) DO (
   REM (Echo "%%g" | FIND /I "2_Optional_Bibliography_Master" 1>NUL) || (
		REM del "%%g"
   REM )
REM )

REM pandoc README.md -o README.docx
REM pandoc -f docx -t gfm AnthonyJamesMcElwee_20211330_FR.docx -o AnthonyJamesMcElwee_20211330_FR.md
REM pandoc -f docx -t markdown_mmd AnthonyJamesMcElwee_20211330_FR.docx -o AnthonyJamesMcElwee_20211330_FR.md
REM pandoc -f docx -t markdown AnthonyJamesMcElwee_20211330_FR.docx -o AnthonyJamesMcElwee_20211330_FR.md
REM pandoc -f docx -t markdown_strict AnthonyJamesMcElwee_20211330_FR.docx -o AnthonyJamesMcElwee_20211330_FR.md
REM pandoc -f docx -t markdown_phpextra AnthonyJamesMcElwee_20211330_FR.docx -o AnthonyJamesMcElwee_20211330_FR.md
REM pandoc -f docx -t commonmark AnthonyJamesMcElwee_20211330_FR.docx -o AnthonyJamesMcElwee_20211330_FR.md

REM pandoc --extract-media=. -s AnthonyJamesMcElwee_20211330_FR.docx -t html -c styles.css -o AnthonyJamesMcElwee_20211330_FR.html
REM powershell -Command "(gc AnthonyJamesMcElwee_20211330_FR.html) -replace './media/', 'file/media/' | Out-File -encoding ASCII AnthonyJamesMcElwee_20211330_FR.html"