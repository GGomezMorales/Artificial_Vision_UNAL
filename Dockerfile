FROM continuumio/miniconda3

RUN conda install jupyter jupyterlab ipykernel -y \
    && conda install -c conda-forge nb_conda_kernels -y \
    && conda install conda-forge::kaggle -y

RUN jupyter-lab --generate-config

COPY ./env.yml /

RUN conda env create -f env.yml

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 unzip -y 

COPY ./.config/jupyter_lab_config.py /root/.jupyter/jupyter_lab_config.py

CMD ["jupyter-lab", "./artificial-vision-project/", "--allow-root"]
