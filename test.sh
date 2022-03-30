#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)

docker volume create atlas2-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
echo ${SCRIPTPATH}
docker run --rm \
        --memory="4g" \
	--init \
        --memory-swap="4g" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/testpreds/:/input/ \
        -v atlas2-output-$VOLUME_SUFFIX:/output/ \
        atlas2

docker run --rm \
        -v atlas2-output-$VOLUME_SUFFIX:/output/ \
        python:3.9-slim cat /output/metrics.json | python -m json.tool

docker volume rm atlas2-output-$VOLUME_SUFFIX
