@echo off
call conda activate solveremf2

REM pandoc README.md -o README.docx
REM pandoc -f docx -t gfm AnthonyJamesMcElwee_20211330_FR.docx -o AnthonyJamesMcElwee_20211330_FR.md
REM pandoc -f docx -t markdown_mmd AnthonyJamesMcElwee_20211330_FR.docx -o AnthonyJamesMcElwee_20211330_FR.md
REM pandoc -f docx -t markdown AnthonyJamesMcElwee_20211330_FR.docx -o AnthonyJamesMcElwee_20211330_FR.md
REM pandoc -f docx -t markdown_strict AnthonyJamesMcElwee_20211330_FR.docx -o AnthonyJamesMcElwee_20211330_FR.md
REM pandoc -f docx -t markdown_phpextra AnthonyJamesMcElwee_20211330_FR.docx -o AnthonyJamesMcElwee_20211330_FR.md
REM pandoc -f docx -t commonmark AnthonyJamesMcElwee_20211330_FR.docx -o AnthonyJamesMcElwee_20211330_FR.md

REM pandoc --extract-media=. -s AnthonyJamesMcElwee_20211330_FR.docx -t html -c styles.css -o AnthonyJamesMcElwee_20211330_FR.html
REM powershell -Command "(gc AnthonyJamesMcElwee_20211330_FR.html) -replace './media/', 'file/media/' | Out-File -encoding ASCII AnthonyJamesMcElwee_20211330_FR.html"

REM pandoc AnthonyJamesMcElwee_20211330_FR.docx -o AnthonyJamesMcElwee_20211330_FR.pdf

echo It is better to load these sub-files into the final DOCX report before PDF conversion but during development they will be updated individually.
REM https://github.com/cognidox/OfficeToPDF/releases
"C:\Users\antho\Downloads\Programmes\officetopdf.exe" "./doc/project_management/AnthonyJamesMcElwee_20211330_PRS.docx" "AnthonyJamesMcElwee_20211330_PRS.pdf"
"C:\Users\antho\Downloads\Programmes\officetopdf.exe" "AnthonyJamesMcElwee_20211330_FR.docx" "AnthonyJamesMcElwee_20211330_FR.pdf"
REM PAUSE