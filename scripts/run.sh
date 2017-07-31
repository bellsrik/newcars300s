#!/bin/bash
spark-submit \
            --class com.cars.bigdata.optimaltimetobuy.RandomForestRegression \
			--master yarn-cluster \
			--num-executors 36 \
            --executor-cores 8 \
            --executor-memory 12G \
            --driver-cores 2 \
            --driver-memory 20G \
            --conf spark.scheduler.listenerbus.eventqueue.size=500000 \
            --conf spark.eventLog.enabled=false \
	        --conf spark.sql.shuffle.partitions=30000 \
            --conf spark.sql.autoBroadcastJoinThreshold=1294000000 \
            --conf spark.akka.timeout=3600 \
            --conf spark.yarn.executor.memoryOverhead=8000 \
            /apps/dev/optimal-time-to-buy/dist/optimal-time-to-buy-assembly-1.0.jar \
			nonprod \
			np1 \
			np2 \
			np3 \
			np4 \
			tree2 \
			depth2 \
            np7