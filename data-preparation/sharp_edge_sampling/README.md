This code has been tested on `Ubuntu 20.04.6 LTS` and requires the following:

* Python 3.10 and PyTorch 2.4.0
* Anaconda (conda3) or Miniconda3
* A CUDA-capable GPU
```shell
sudo apt-get update &&  sudo apt-get install -y ninja-build libxi6
conda create -n Dora python=3.10 -y
conda activate Dora
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/ashawkey/cubvh@4c0de5fc7f9f836fd9c562616bdfeb77a942ba5d
pip install diso 
pip install -r requirements.txt
```

**Overview**

The data processing workflow consists of two steps. The first step is to convert non-watertight models into watertight ones. The second step is to perform sampling on the watertight models. For some reasons, we won't release the code for the first step for now. However, we've provided an alternative solution, and its effect won't differ much from what we actually used. If your model is already watertight, you can directly proceed to the second step. Note that the .obj files in Dora-bench have already been converted into watertight models.

**Step 1: (GPU) Convert non-watertight models into watertight ones**
```shell
python detect_path.py   --directory_to_search ./Objaverse \
                        --json_file_path ./mesh_path.json  \
                        --file_type .glb
```
Use the following comment for Dora-VAE v1.1
```shell
python to_watertight_mesh.py  --resolution 256 \
                              --json_file_path ./mesh_path.json \
                              --remesh_target_path ./remesh
```
Use the following comment for Dora-VAE v1.2 or for finetuning Dora-VAE v1.1.
```shell
python to_watertight_mesh.py  --resolution 512 \
                              --json_file_path ./mesh_path.json \
                              --remesh_target_path ./remesh
```

**Step 2: (CPU-Only) Perform sharp edge sampling on the watertight models**
```shell
python detect_path.py   --directory_to_search ./remesh \
                        --json_file_path ./watertight_path.json \
                        --file_type .obj
python sharp_sample.py  --json_file_path ./watertight_path.json  \
                        --point_number 65536 \
                        --angle_threshold 15 \
                        --sharp_point_path ./sharp_point_ply \
                        --sample_path ./sample
```

## Acknowledgement

- [cubvh](https://github.com/ashawkey/cubvh) provides a fast implementation to compute udf.
- [pysdf](https://github.com/sxyu/sdf) provides a fast implementation to compute sdf.
- [fpsample](https://github.com/leonardodalinky/fpsample) provides a fast implementation to compute fps
- [diso](https://github.com/SarahWeiii/diso) provides a fast implementation to extract iso-surface.
- [vscode-mesh-viewer](https://github.com/ashawkey/vscode-mesh-viewer) provides a plugin for previewing meshes in VSCode.