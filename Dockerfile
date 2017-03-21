# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

FROM newreg.creditx.com/common/scipy-notebook

USER root
RUN echo "deb http://httpredir.debian.org/debian jessie-backports main" >> /etc/apt/sources.list
RUN sed -i "s/httpredir\.debian\.org/mirrors\.ustc\.edu\.cn/g" /etc/apt/sources.list

RUN echo "channels:\n\
  - http://newreg.creditx.com:10443/pkgs/creditx\n\
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free" > ~/.condarc

RUN apt-get update && apt-get install -y  -t jessie-backports ca-certificates-java openjdk-8-jre-headless openjdk-8-jdk-headless openjdk-8-jre openjdk-8-jdk tree vim && \
    apt-get clean  && \
    rm -rf /var/lib/apt/lists/*

USER $NB_USER
COPY .jupyter/jupyter_notebook_config.py /home/$NB_USER/.jupyter

RUN mkdir -p ~/.ipython/profile_default/startup
RUN echo "from doit.tools import register_doit_as_IPython_magic\n\
register_doit_as_IPython_magic()\n\
%load_ext sql\n\
%load_ext jupyter_cms\n" >  ~/.ipython/profile_default/startup/startup_magic.ipy

RUN conda install --force --yes jupyter_nbextensions_configurator
