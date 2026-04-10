# syntax=docker/dockerfile:1
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CONDA_ENV_DIR=/opt/conda/envs/.atmos_25

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
    curl \
    bzip2 \
    tar \
    gzip \
    git \
    build-essential \
    python3-dev \
    gfortran \
    pkg-config \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

ARG NB_USER=worker
ARG NB_UID=1000
ARG NB_GID=1000

RUN groupadd -g ${NB_GID} ${NB_USER} || true \
 && useradd -m -s /bin/bash -u ${NB_UID} -g ${NB_GID} ${NB_USER}

WORKDIR /home/${NB_USER}

COPY atmos_25_condapack.tar.gz /atmos_25_condapack.tar.gz

RUN mkdir -p ${CONDA_ENV_DIR} \
 && tar -xzf /atmos_25_condapack.tar.gz -C ${CONDA_ENV_DIR} \
 && rm /atmos_25_condapack.tar.gz \
 && if [ -x "${CONDA_ENV_DIR}/bin/conda-unpack" ]; then "${CONDA_ENV_DIR}/bin/python" "${CONDA_ENV_DIR}/bin/conda-unpack"; fi \
 && chown -R ${NB_USER}:${NB_USER} ${CONDA_ENV_DIR} /home/${NB_USER}

ENV PATH="${CONDA_ENV_DIR}/bin:${PATH}"

USER ${NB_USER}
WORKDIR /home/${NB_USER}

CMD ["/bin/bash", "-l"]