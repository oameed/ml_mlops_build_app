# syntax=docker/dockerfile:1.7-labs

FROM continuumio/miniconda3:latest

ARG path_to_prj=/home/ml_mlops_build_app
ARG conda_env_file=conda-tf2py.yml
ARG conda_env_path=/opt/conda/envs/tf2py/bin/python

COPY           $conda_env_file   $path_to_prj/
COPY           predict.py        $path_to_prj/
COPY --parents run/*/checkpoints $path_to_prj/
COPY --parents run/*/predictions $path_to_prj/

RUN conda env create --file $path_to_prj/$conda_env_file
RUN ln -s $conda_env_path /usr/local/bin/path_to_python
RUN rm $path_to_prj/$conda_env_file

CMD ["/bin/bash"]

