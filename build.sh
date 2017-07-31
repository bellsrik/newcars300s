#!/usr/bin/env sh

set -e

docker run -v $(pwd):/root hseeberger/scala-sbt sbt clean assembly
