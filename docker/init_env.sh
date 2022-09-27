#!/bin/bash
set -euxo pipefail

if [ ! -f .env ]; then
  cat <<EOT > .env
  HOST_UID=`id -u`
  HOST_GID=`id -g`
EOT
fi

