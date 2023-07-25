@echo off

set "folder=.\instances_output\"

for %%F in ("%folder%\*_o*.npy") do (
    echo Deleting "%%~nxF"
    del "%%F"
)
