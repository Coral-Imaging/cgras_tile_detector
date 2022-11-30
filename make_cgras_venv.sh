# script to automatically make conda environment defined in cgras.yml
mamba env create -f cgras.yml

# not sure how to get these to work after-the-fact
conda activate cgras
pip3 install rawpy pupil_apriltags
