# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

FROM registry.creditx.com:5000/scipy-notebook

USER root
RUN sed -i "s/httpredir\.debian\.org/mirrors4\.tuna\.tsinghua\.edu\.cn/g" /etc/apt/sources.list


RUN apt-get update && apt-get install -y openjdk-7-jdk tree vim && \
    apt-get clean  && \
    rm -rf /var/lib/apt/lists/*

RUN conda install --yes pytest scikit-optimize 
RUN bash -c "source activate python2 && \
     conda install pytest scikit-optimize && \
          . deactivate"
