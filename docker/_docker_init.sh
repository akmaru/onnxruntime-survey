#!/usr/bin/env bash

USER=developer

# overwrite uid and gid
usermod -u $HOST_UID ${USER}
groupmod -g $HOST_GID ${USER}

HOME_DIR=/home/${USER}
BASHRC_PATH=${HOME_DIR}/.bashrc

# keep some environments
echo "export PYTHONPATH=${PYTHONPATH}" >> ${BASHRC_PATH}
echo "export PYTHONIOENCODING=utf-8" >> ${BASHRC_PATH}
echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> ${BASHRC_PATH}
echo "export PATH=${PATH}" >> ${BASHRC_PATH}
echo "cd /workspaces/onnxruntime-survey" >> ${BASHRC_PATH}

# change to the developer
chown ${USER}:${USER} -R $HOME_DIR
su - ${USER}

