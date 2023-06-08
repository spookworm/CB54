eval "$('C:/ProgramData/anaconda3/Scripts/conda.exe' 'shell.bash' 'hook')"
conda activate base
conda env export -n solveremf2 > solveremf2.yml
read -p "Press enter to continue"