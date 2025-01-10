## Install ComfyUI-SAM2-Realtime
Clone the repo
```
cd custom_nodes
git clone https://github.com/pschroedl/ComfyUI-SAM2-Realtime.git
cd ComfyUI-SAM2-Realtime
```

Install requirements
#### Linux
```
pip install -r requirements.txt
pip install .
```

#### Windows**
Use the `--no-build-isolation` flag when using conda on Windows to preserve env variables 
```
pip install -r requirements.txt
pip install . --no-build-isolation
```
