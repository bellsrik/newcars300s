#!/bin/bash
spark-submit \
            --class com.cars.bigdata.optimaltimetobuy.GBTRegression \
			c:/Users/sbellary/ot2b/optimal-time-to-buy/target/scala-2.10/optimal-time-to-buy-assembly-0.2.3.jar \
			local \
			l1 \
			l2 \
			l3 \
			iter1 \
			depth1