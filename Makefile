create_env:
	conda create -n nml_38 python=3.8
	mamba install -n nml_38 matplotlib numpy scipy tqdm seaborn pandas flake8 black scikit-learn
	mamba install -n nml_38 -c conda-forge jupyterlab

sync: sync_data_files sync_experiments sync_notebooks sync_scripts
