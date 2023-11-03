'''
Simple web service wrapping a Word2Vec as implemented in Gensim
Example call: curl http://127.0.0.1:5000/word2vec/n_similarity?ws1=sushi&ws1=shop&ws2=japanese&ws2=restaurant
@TODO: Add more methods
@TODO: Add command line parameter: path to the trained model
@TODO: Add command line parameters: host and port
'''
from __future__ import print_function

from future import standard_library
standard_library.install_aliases()
from builtins import str
from flask import Flask, request, jsonify
from flask_restful import Resource, Api, reqparse
import gensim.models.keyedvectors as word2vec
from gensim import utils, matutils
from numpy import exp, dot, zeros, outer, random, dtype, get_include, float32 as REAL,\
     uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis, ndarray, empty, sum as np_sum
import pickle
import argparse
import base64
import sys
from datetime import datetime

parser = reqparse.RequestParser()


def filter_words(words):
    #print("here is model vocab",flush=True)
    #print(model.vocab,flush=True)

    print("words are here",flush=True)
    print(words,flush=True)

    if words is None:
        return
    return [word for word in words if word in model.vocab]


class N_Similarity(Resource):
    def get(self):
        print("this is before the parser",flush=True)
        parser = reqparse.RequestParser()
        print("this is after the parser is obtained",flush=True)
        parser.add_argument('ws1', type=str, required=True, help="Word set 1 cannot be blank!", action='append',location='args')
        parser.add_argument('ws2', type=str, required=True, help="Word set 2 cannot be blank!", action='append',location='args')
        #get the arguments and conver to lower case for English
        args = parser.parse_args()
        
        print("here are the args",flush=True)
        print(args,flush=True)

        ws1=args['ws1'] #list of ws1 object
        ws2=args['ws2'] #list of ws2 object
        ws1lower= [x.lower() for x in ws1]
        ws2lower= [x.lower() for x in ws2]

        print("here are the final args",flush=True)
        print(ws1,ws2,ws1lower,ws2lower,flush=True)

        filter1=filter_words(ws1lower)
        filter2=filter_words(ws2lower)

        print("filter1",flush=True)
        print(filter1,flush=True)
        print("filter2",flush=True)
        print(filter2,flush=True)

        print("before calculating similarity",flush=True)
        
        if not filter1 or not filter2:
            return 0

        res=model.n_similarity(filter1,filter2).item()

        #res= model.n_similarity(filter_words(args['ws1']),filter_words(args['ws2'])).item()
        print("after calculating similarity",flush=True)
        #return (args,filter1,filter2) 
        return res
        #return model.n_similarity(filter_words(args['ws1']),filter_words(args['ws2'])).item()


class Similarity(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('w1', type=str, required=True, help="Word 1 cannot be blank!")
        parser.add_argument('w2', type=str, required=True, help="Word 2 cannot be blank!")
        args = parser.parse_args()
        return model.similarity(args['w1'], args['w2']).item()


class MostSimilar(Resource):
    def get(self):
        if (norm == "disable"):
            return "most_similar disabled", 400
        parser = reqparse.RequestParser()
        parser.add_argument('positive', type=str, required=False, help="Positive words.", action='append')
        parser.add_argument('negative', type=str, required=False, help="Negative words.", action='append')
        parser.add_argument('topn', type=int, required=False, help="Number of results.")
        args = parser.parse_args()
        pos = filter_words(args.get('positive', []))
        neg = filter_words(args.get('negative', []))
        t = args.get('topn', 10)
        pos = [] if pos == None else pos
        neg = [] if neg == None else neg
        t = 10 if t == None else t
        print("positive: " + str(pos) + " negative: " + str(neg) + " topn: " + str(t),flush=True)
        try:
            res = model.most_similar_cosmul(positive=pos,negative=neg,topn=t)
            return res
        except Exception as e:
            print(e,flush=True)
            print(res,flush=True)


class Model(Resource):
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('word', type=str, required=True, help="word to query.")
        args = parser.parse_args()
        try:
            res = model[args['word']]
            res = base64.b64encode(res).decode()
            return res
        except Exception as e:
            print(e,flush=True)
            return

class ModelWordSet(Resource):
    def get(self):
        try:
            res = base64.b64encode(pickle.dumps(set(model.index2word))).decode()
            return res
        except Exception as e:
            print(e,flush=True)
            return
class TestURL(Resource):
    def get(self):
        try:
            res="hi you get this URL"
            return res
        except Exception as e:
            print(e,flush=True)
            return "oops you did not get this URL"

app = Flask(__name__)
api = Api(app)

@app.errorhandler(404)
def pageNotFound(error):
    return "page not found"

@app.errorhandler(500)
def raiseError(error):
    return error

if __name__ == '__main__':
    global model
    global norm

    #----------- Parsing Arguments ---------------
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="Path to the trained model")
    p.add_argument("--binary", help="Specifies the loaded model is binary")
    p.add_argument("--host", help="Host name (default: localhost)")
    p.add_argument("--port", help="Port (default: 5000)")
    p.add_argument("--path", help="Path (default: /word2vec)")
    p.add_argument("--norm", help="How to normalize vectors. clobber: Replace loaded vectors with normalized versions. Saves a lot of memory if exact vectors aren't needed. both: Preserve the original vectors (double memory requirement). already: Treat model as already normalized. disable: Disable 'most_similar' queries and do not normalize vectors. (default: both)")
    args = p.parse_args()

    model_path = args.model if args.model else "./model.bin.gz"
    binary = True if args.binary else False
    host = args.host if args.host else "localhost"
    path = args.path if args.path else "/word2vec"
    port = int(args.port) if args.port else 5000
    if not args.model:
        print("Usage: word2vec-api.py --model path/to/the/model [--host host --port 1234]")
    
    print("Loading model...")
    # datetime object containing current date and time
    now = datetime.now()
    print("now =", now)
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("date and time =", dt_string)
    model = word2vec.KeyedVectors.load_word2vec_format(model_path, binary=binary)

    now=datetime.now()
    dt_string2 = now.strftime("%d/%m/%Y %H:%M:%S")
    print("model load finished");
    print("date and time =", dt_string2)

    norm = args.norm if args.norm else "both"
    norm = norm.lower()
    if (norm in ["clobber", "replace"]):
        norm = "clobber"
        print("Normalizing (clobber)...")
        model.init_sims(replace=True)
    elif (norm == "already"):
        model.wv.vectors_norm = model.wv.vectors  # prevent recalc of normed vectors (model.syn0norm = model.syn0)
    elif (norm in ["disable", "disabled"]):
        norm = "disable"
    else:
        norm = "both"
        print("Normalizing...")
        model.init_sims()
    if (norm == "both"):
        print("Model loaded.")
    else:
        print("Model loaded. (norm=",norm,")")

    print("here is path:")
    print(path)
    
    api.add_resource(N_Similarity, path+'/n_similarity')
    api.add_resource(Similarity, path+'/similarity')
    api.add_resource(MostSimilar, path+'/most_similar')
    api.add_resource(Model, path+'/model')
    api.add_resource(ModelWordSet, '/word2vec/model_word_set')
    api.add_resource(TestURL,'/word2vec/testURL')
    app.run(host=host, port=port)
