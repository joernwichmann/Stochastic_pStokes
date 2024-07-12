This repository contains code that was used for the data generation of the article:
	https://arxiv.org/abs/2307.13253

SETTING UP THE SIMULATION:
In order to run the simulations one first needs to install the finite element package
	FIREDRAKE	https://www.firedrakeproject.org/
Its installation should be done with the additional keyword '--netgen' to include the netgen package. 
After successful installation, start the virtual environment provided by firedrake:
	Navigate to the firedrake directory and use "source bin/activate" in the terminal. 
Next, install the package 	
	TQDM		https://tqdm.github.io/
This can be done by running "pip install tqdm"
The setup is complete and the virtual enviroment can be deactivated by running "deactivate"


RUNNING THE SIMULATION:
We first need to start firedrake's virtual environment:
	Navigate to the firedrake directory and use "source bin/activate" in the terminal. 
Now, navigate to the directory 'Stochastic_pStokes' and run the following to start the simulation
	python3 run_{TH15,TH3,SV15,SV3}.py
	
