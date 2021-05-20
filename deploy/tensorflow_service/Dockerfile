FROM nvidia/cuda:11.0-cudnn8-runtime-ubuntu16.04 as nvidia-cuda

FROM bentoml/model-server:0.12.1-py38

COPY --from=nvidia-cuda /usr/local/cuda-11.0 /usr/local/cuda
COPY --from=nvidia-cuda /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda/lib64/
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64

# Configure PIP install arguments, e.g. --index-url, --trusted-url, --extra-index-url
ARG EXTRA_PIP_INSTALL_ARGS=
ENV EXTRA_PIP_INSTALL_ARGS $EXTRA_PIP_INSTALL_ARGS

ARG UID=1034
ARG GID=1034
RUN groupadd -g $GID -o bentoml && useradd -m -u $UID -g $GID -o -r bentoml

ARG BUNDLE_PATH=/home/bentoml/bundle
ENV BUNDLE_PATH=$BUNDLE_PATH
ENV BENTOML_HOME=/home/bentoml/

RUN mkdir $BUNDLE_PATH && chown bentoml:bentoml $BUNDLE_PATH -R
WORKDIR $BUNDLE_PATH

# copy over the init script; copy over entrypoint scripts
COPY --chown=bentoml:bentoml bentoml-init.sh docker-entrypoint.sh ./
RUN chmod +x ./bentoml-init.sh

# Copy docker-entrypoint.sh again, because setup.sh might not exist. This prevent COPY command from failing.
COPY --chown=bentoml:bentoml docker-entrypoint.sh setup.s[h] ./
RUN ./bentoml-init.sh custom_setup

COPY --chown=bentoml:bentoml docker-entrypoint.sh python_versio[n] ./
RUN ./bentoml-init.sh ensure_python

COPY --chown=bentoml:bentoml environment.yml ./
RUN ./bentoml-init.sh restore_conda_env

COPY --chown=bentoml:bentoml requirements.txt ./
RUN ./bentoml-init.sh install_pip_packages

COPY --chown=bentoml:bentoml docker-entrypoint.sh bundled_pip_dependencie[s]  ./bundled_pip_dependencies/
RUN rm ./bundled_pip_dependencies/docker-entrypoint.sh && ./bentoml-init.sh install_bundled_pip_packages

# copy over model files
COPY --chown=bentoml:bentoml . ./

# the env var $PORT is required by heroku container runtime
ENV PORT 5000
EXPOSE $PORT

USER bentoml
RUN chmod +x ./docker-entrypoint.sh
ENTRYPOINT [ "./docker-entrypoint.sh" ]
CMD ["bentoml", "serve-gunicorn", "./"]