# How to set up the project
- Install Anaconda
- Go to the `project` directory
- Create an Anaconda environment
```
conda create -n <ENV-NAME> -c conda-forge jupyterlab=3.6.3 "ipykernel>=6" xeus-python
```
- Activate the environment
```
conda activate <ENV-NAME>
```
- Install dependencies
```
pip install -r requirements.txt
```
- Run the Jupyter server
```
jupyter lab
```