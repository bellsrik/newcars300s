#!/bin/bash
spark-submit \
            --class com.cars.bigdata.optimaltimetobuy.GBTRegression \
			--master yarn-cluster \
			--num-executors 36 \
            --executor-cores 5 \
            --executor-memory 8G \
            --driver-cores 2 \
            --driver-memory 20G \
            --conf spark.shuffle.io.numConnectionsPerPeer=1 \
            --conf spark.shuffle.sort.bypassMergeThreshold=1000 \
            --conf spark.shuffle.consolidateFiles=true \
            --conf spark.shuffle.file.buffer=2048k \
            --conf spark.yarn.executor.memoryOverhead=8000 \
            --conf spark.speculation=true \
            /apps/dev/optimal-time-to-buy/dist/optimal-time-to-buy-assembly-1.0.jar \
			nonprod \
			np1 \
			np2 \
			np3 \
			iter2 \
			depth2