# STAGE 1: Base system Linux Mint 21.3
FROM linuxmintd/mint21.3-amd64 AS base

# System settings
RUN ln -snf /usr/share/zoneinfo/Europe/Moscow /etc/localtime && \
    echo Europe/Moscow > /etc/timezone

# System dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        wget git build-essential ca-certificates libgl1 libglib2.0-0 sudo curl && \
    apt-get autoremove -y && apt-get autoclean && \
    rm -rf /var/lib/apt/lists/*

# STAGE 2: Install CUDA for GTX 980 Maxwell
FROM base AS cuda

ENV TORCH_CUDA_ARCH_LIST="5.2 5.2+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV TORCH_USE_CUDA_DSA="1"
ENV FORCE_CUDA="1"
ENV CUDA_HOME="/usr/local/cuda-12.4"

# CUDA installation
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y cuda-toolkit-12-4 && \
    rm -rf /var/lib/apt/lists/*

# STAGE 3: Miniconda setup
FROM cuda AS miniconda

# Create non-root user
RUN useradd -m user && \
    mkdir -p /workspace && \
    chown -R user:user /workspace && \
    usermod -aG sudo user && \
    echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER user
WORKDIR /workspace

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -u -p /home/user/miniconda3 && \
    rm miniconda.sh

ENV PATH="/home/user/miniconda3/bin:$PATH"

RUN conda update -n base -c defaults conda -y && \
    conda update --all -y && \
    conda init bash

ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# STAGE 4: Core Python environment (where shell activation begins)
FROM miniconda AS core_env

## Use the login shell to ensure .bashrc is loaded
SHELL ["/bin/bash", "-l", "-c"]

# Create and activate environment
RUN conda create -y -n py39 python=3.9 && \
    echo "conda activate py39" >> ~/.bashrc

# Create .local/bin directory
RUN mkdir -p ~/.local/bin

# Install Poetry with virtualenvs disabled
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && \
    source ~/.bashrc && \
    export PATH="$HOME/.local/bin:$PATH" && \
    poetry config virtualenvs.create false && \
    poetry self update && \
    poetry completions bash > $HOME/.bash_completion && \
    echo 'source $HOME/.bash_completion' >> ~/.bashrc

#STAGE 5: torch and numpy
FROM core_env AS torch_deps

SHELL ["/bin/bash", "-l", "-c"]

# Install Numpy 1.26.4 and openmim for MM-based apps
RUN conda run -n py39 pip install numpy==1.26.4 && \
    conda run -n py39 pip install -U openmim

RUN conda run -n py39 pip install torch==2.5.1 torchvision==0.20.1 pytorch-lightning

# STAGE 6: Model-specific dependencies
FROM torch_deps AS app_deps

SHELL ["/bin/bash", "-l", "-c"]

RUN cd /workspace && \
    git clone https://github.com/fatrybl/pcd-image-model.git && \
    cd pcd-image-model && \
    conda run -n py39 poetry install --verbose

# STAGE 7: Dependencies for MM libs
FROM app_deps AS mm_deps

SHELL ["/bin/bash", "-l", "-c"]

# Install MMEngine, MMCV, MMDetection, and MMDetection3D
RUN conda run -n py39 mim install mmengine && \
    conda run -n py39 mim install mmcv==2.1 mmdet==3.2 mmdet3d

# Install TorchEx
RUN cd /tmp && git clone https://github.com/Abyssaledge/TorchEx.git && \
    cd TorchEx && \
    conda run -n py39 pip install --no-build-isolation . && \
    cd .. && \
    rm -rf TorchEx

# STAGE 8: Final clean up
FROM mm_deps AS final

#SHELL ["/bin/bash", "-l", "-c"]

# Cleanup
RUN conda clean -afy && \
    sudo rm -rf \
      ~/.cache/pip \
      ~/.cache/pypoetry \
      ~/miniconda3/pkgs/* \
      /tmp/* /var/tmp/* \
      /var/lib/apt/lists/* \
      /var/cache/apt/archives/*

CMD ["/bin/bash"]