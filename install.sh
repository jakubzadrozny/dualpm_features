# /bin/bash
conda install -c conda-forge gcc_linux-64=9 gxx_linux-64=9 -y > out.log 2>&1 && \
conda install -c "nvidia/label/cuda-11.6.1" libcusolver-dev -y > out.log 2>&1 && \
conda install pytorch=1.13.1 torchvision=0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y > out.log 2>&1 && \
uv pip install ninja==1.10 --no-build-isolation > out.log 2>&1 && \
uv pip install timm==0.6.11 xformers==0.0.16 "pillow==9.2.*" "numpy==1.*" --no-build-isolation > out.log 2>&1 && \
uv pip install "https://github.com/facebookresearch/detectron2/archive/v0.6.zip" --no-build-isolation > out.log 2>&1 && \
uv pip install "git+https://github.com/Junyi42/sd-dino.git#subdirectory=third_party/Mask2Former" --no-build-isolation > out.log 2>&1 && \
uv pip install "git+https://github.com/Junyi42/sd-dino.git#subdirectory=third_party/ODISE" --no-build-isolation > out.log 2>&1 && \
uv pip install -e . --no-build-isolation > out.log 2>&1
