#!/usr/bin/env bash

./build.sh

docker save atlas2 | gzip -c > atlas2.tar.gz
