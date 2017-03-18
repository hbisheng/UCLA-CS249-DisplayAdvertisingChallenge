import datetime
import sys
import shutil
import hashlib

from collections import defaultdict
from random import randint
from math import exp, log, sqrt

from pyspark import SparkConf,SparkContext
from pyspark.sql import SQLContext

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import *
from pyspark.mllib.linalg import SparseVector

# codes borrowed from PYMMH3 at https://github.com/wc-duck/pymmh3
def murmur3(key, seed = 0x0 ):
    ''' implements 32bit murmur3 hash. '''

    key = bytearray( key)

    def fmix( h ):
        h ^= h >> 16
        h  = ( h * 0x85ebca6b ) & 0xFFFFFFFF
        h ^= h >> 13
        h  = ( h * 0xc2b2ae35 ) & 0xFFFFFFFF
        h ^= h >> 16
        return h;

    length = len( key )
    nblocks = int( length / 4 )

    h1 = seed;

    c1 = 0xcc9e2d51
    c2 = 0x1b873593

    # body
    for block_start in xrange( 0, nblocks * 4, 4 ):
        k1 = key[ block_start + 3 ] << 24 | \
             key[ block_start + 2 ] << 16 | \
             key[ block_start + 1 ] <<  8 | \
             key[ block_start + 0 ]

        k1 = c1 * k1 & 0xFFFFFFFF
        k1 = ( k1 << 15 | k1 >> 17 ) & 0xFFFFFFFF # inlined ROTL32
        k1 = ( c2 * k1 ) & 0xFFFFFFFF;

        h1 ^= k1
        h1  = ( h1 << 13 | h1 >> 19 ) & 0xFFFFFFFF # inlined _ROTL32
        h1  = ( h1 * 5 + 0xe6546b64 ) & 0xFFFFFFFF

    # tail
    tail_index = nblocks * 4
    k1 = 0
    tail_size = length & 3

    if tail_size >= 3:
        k1 ^= key[ tail_index + 2 ] << 16
    if tail_size >= 2:
        k1 ^= key[ tail_index + 1 ] << 8
    if tail_size >= 1:
        k1 ^= key[ tail_index + 0 ]

    if tail_size != 0:
        k1  = ( k1 * c1 ) & 0xFFFFFFFF
        k1  = ( k1 << 15 | k1 >> 17 ) & 0xFFFFFFFF # _ROTL32
        k1  = ( k1 * c2 ) & 0xFFFFFFFF
        h1 ^= k1

    return fmix( h1 ^ length )

def normal_hash(key):
    return int(key, 16)
    
def parsePoint(row):
    feas = [0]
    i = 0
    for i in range(0, 39):
        """ Choose the hash function """
        
        """ SimpleHash"""
        #hash_val = normal_hash(row[i + 1] + keys[i][1:]) % BUCKET_SIZE
        
        """ MurmurHash """
        #hash_val = murmur3( (row[i + 1] + '=' + keys[i][1:]).encode("utf-8") ) % BUCKET_SIZE
        
        """ MD5Hash"""
        hash_val = int(hashlib.md5((keys[i][1:] + row[i + 1]).encode('utf8')).hexdigest(), 16) % (BUCKET_SIZE - 1) + 1

        feas.append(hash_val)

    return (int(row[0]), feas) # label and feature


def get_probability(feas, model):
    wTx = 0.
    for i in feas:
        wTx += model[i] * 1.
    return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))


def logloss(prob, label):
    prob = max(min(prob, 1. - 10e-12), 10e-12)
    return -log(prob) if label == 1. else -log(1. - prob)

STEP = .1  
BUCKET_SIZE = 2 ** 20
keys = ['I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26']


def train_batch(points_iter):
    """ train a logistic regression model within an RDD partition"""

    batch_model = [0.] * BUCKET_SIZE
    batch_cnt   = [0.] * BUCKET_SIZE
    
    for point in points_iter:
        label = point[0]
        features = point[1]
        
        p = get_probability(features, batch_model)
        
        for i in features:
            batch_model[i] -= (p - label) * STEP / (sqrt(batch_cnt[i]) + 1.)
            batch_cnt[i] += 1
        
    return [(batch_model, batch_cnt)]

# combine two model based on the feature frequency
def combine_model(model1_cnt, model2_cnt):
    model1 = model1_cnt[0]
    cnt1   = model1_cnt[1]
    
    model2 = model2_cnt[0]
    cnt2   = model2_cnt[1]    
    
    for i in range(0, BUCKET_SIZE):
        model1[i] = (cnt1[i] * model1[i] + cnt2[i] * model2[i]) / (cnt1[i] + cnt2[i]) if(cnt1[i] + cnt2[i] != 0) else 0.
        cnt1[i] += cnt2[i]

    return (model1, cnt1)

def main():

    PREFIX = './'
    
    TRAIN_INPUT_FILE = PREFIX + 'train_100k.txt'
    LOGLOSS_OUTPUT_FILE = PREFIX + 'output'

    conf = SparkConf().setAppName("LogisticClassifer")
    sc = SparkContext(conf = conf)
    
    train_parsed = sc.textFile(TRAIN_INPUT_FILE).map( lambda p: p.split('\t') ).map(parsePoint)
    
    ratio = [0.9, 0.1]

    run = []

    train_portion, test_portion = train_parsed.randomSplit(ratio)

    (final_weights, final_cnt) = train_portion.mapPartitions(train_batch).reduce(combine_model)

    test_avglogloss = test_portion.map(lambda point: logloss(get_probability(point[1], final_weights), point[0]) ).sum() / test_portion.count()
        
    print test_avglogloss

    run.append(test_avglogloss)
    run.append(train_portion.getNumPartitions())
    run.append(final_weights)
    
    shutil.rmtree(LOGLOSS_OUTPUT_FILE + '_' + str(ratio), ignore_errors=True)
    sc.parallelize(run).repartition(1).saveAsTextFile(LOGLOSS_OUTPUT_FILE + '_' + str(ratio)) 
    sc.stop()

if __name__ == "__main__": main()