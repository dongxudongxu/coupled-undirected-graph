# Analytical engine

ARG REGISTRY=registry.cn-hongkong.aliyuncs.com
ARG BUILDER_VERSION=latest
ARG RUNTIME_VERSION=latest
############### BUILDER: ANALYTICAL #######################
FROM $REGISTRY/graphscope/graphscope-dev:$BUILDER_VERSION AS builder

ARG CI=false

COPY --chown=graphscope:graphscope . /home/graphscope/GraphScope

RUN cd /home/graphscope/GraphScope/ && \
    if [ "${CI}" = "true" ]; then \
        cp -r artifacts/analytical /home/graphscope/install; \
    else \
        export INSTALL_DIR=/home/graphscope/install; \
        mkdir ${INSTALL_DIR}; \
        . /home/graphscope/.graphscope_env; \
        make analytical-install INSTALL_PREFIX=${INSTALL_DIR}; \
        strip ${INSTALL_DIR}/bin/grape_engine; \
        strip ${INSTALL_DIR}/lib/*.so; \
        sudo cp -rs ${INSTALL_DIR}/* ${GRAPHSCOPE_HOME}/; \
        python3 ./k8s/utils/precompile.py --graph --output_dir ${INSTALL_DIR}/builtin; \
        strip ${INSTALL_DIR}/builtin/*/*.so || true; \
    fi

############### RUNTIME: ANALYTICAL #######################
FROM $REGISTRY/graphscope/vineyard-dev:$RUNTIME_VERSION AS analytical

ENV GRAPHSCOPE_HOME=/opt/graphscope
ENV PATH=$PATH:$GRAPHSCOPE_HOME/bin LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GRAPHSCOPE_HOME/lib

USER root

COPY ./k8s/utils/kube_ssh /usr/local/bin/kube_ssh
COPY --from=builder /home/graphscope/install /opt/graphscope/
RUN mkdir -p /tmp/gs && (mv /opt/graphscope/builtin /tmp/gs/builtin || true) && chown -R graphscope:graphscope /tmp/gs
RUN chmod +x /opt/graphscope/bin/*

# RUN apt-get update -y && apt-get install openssh-server dnsutils -y && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/run/sshd

RUN sed -i 's/[ #]\(.*StrictHostKeyChecking \).*/ \1no/g' /etc/ssh/ssh_config && \
    echo "    UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    echo "StrictModes no" > /etc/ssh/sshd_config && \
    echo "PidFile /tmp/sshd.pid" >> /etc/ssh/sshd_config && \
    echo "HostKey ~/.ssh/id_rsa" >> /etc/ssh/sshd_config && \
    echo "Port 2222" >> /etc/ssh/sshd_config && \
    echo "ListenAddress 0.0.0.0" >> /etc/ssh/sshd_config && \
    sed -i 's/.*Port 22.*/   Port 2222/g' /etc/ssh/ssh_config

USER graphscope
WORKDIR /home/graphscope

COPY ./k8s/dockerfiles/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

############### BUILDER: ANALYTICAL-JAVA #######################
FROM $REGISTRY/graphscope/graphscope-dev:$BUILDER_VERSION AS builder-java

COPY --chown=graphscope:graphscope . /home/graphscope/GraphScope

RUN cd /home/graphscope/GraphScope/ && \
    if [ "${CI}" = "true" ]; then \
        cp -r artifacts/analytical-java /home/graphscope/install; \
    else \
        export INSTALL_DIR=/home/graphscope/install; \
        mkdir ${INSTALL_DIR}; \
        . /home/graphscope/.graphscope_env; \
        make analytical-java-install INSTALL_PREFIX=${INSTALL_DIR}; \
        strip ${INSTALL_DIR}/bin/grape_engine; \
        strip ${INSTALL_DIR}/lib/*.so; \
        sudo cp -rs ${INSTALL_DIR}/* ${GRAPHSCOPE_HOME}/; \
        python3 ./k8s/utils/precompile.py --graph --output_dir ${INSTALL_DIR}/builtin; \
        strip ${INSTALL_DIR}/builtin/*/*.so || true; \
    fi

############### RUNTIME: ANALYTICAL-JAVA #######################

FROM vineyardcloudnative/manylinux-llvm:2014-11.0.0 AS llvm

FROM $REGISTRY/graphscope/vineyard-dev:$RUNTIME_VERSION AS analytical-java
COPY --from=llvm /opt/llvm11.0.0 /opt/llvm11
ENV LLVM11_HOME=/opt/llvm11
ENV LIBCLANG_PATH=$LLVM11_HOME/lib LLVM_CONFIG_PATH=$LLVM11_HOME/bin/llvm-config

ENV GRAPHSCOPE_HOME=/opt/graphscope
ENV PATH=$PATH:$GRAPHSCOPE_HOME/bin LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$GRAPHSCOPE_HOME/lib

USER root
COPY ./k8s/utils/kube_ssh /usr/local/bin/kube_ssh
COPY --from=builder-java /home/graphscope/install /opt/graphscope/
RUN mkdir -p /tmp/gs && (mv /opt/graphscope/builtin /tmp/gs/builtin || true) && chown -R graphscope:graphscope /tmp/gs
RUN chmod +x /opt/graphscope/bin/*

USER graphscope
WORKDIR /home/graphscope

ENV PATH=${PATH}:/home/graphscope/.local/bin
