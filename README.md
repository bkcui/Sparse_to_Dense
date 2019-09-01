# Sparse_to_Dense
Reproduce of Sparse to Dense paper by Fangchang Ma.

cuDNN version 9.2, Python version 3.6, Pytorch version 1.2

Data and Dataloader from https://github.com/XinJCheng/CSPN

-Requirements
	'''sudo apt-get update
	sudo apt-get install -y libhdf5-serial-dev hdf5-tools
	pip3 install h5py pandas matplotlib imageio opencv-python'''

NYU dataset
	```bash
	mkdir data; cd data
	wget http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz
	tar -xvf nyudepthv2.tar.gz && rm -f nyudepthv2.tar.gz
    mv nyudepthv2 nyudepth_hdf5```
