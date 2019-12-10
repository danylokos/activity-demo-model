#!/usr/bin/env python

'''
./converter.py \
	--model model.h5 \
	--desc "Predicts either a phone is in the hands or in a pocket" \
	--input_desc "Sensor samples (acc, gyro, mag, 50Hz)" \
	--output_desc "1 - phone in the hands, 0 - phone in a pocket" \
	--author "Danylo Kostyshyn" \
	--license="MIT"
'''

from __future__ import absolute_import
import os
import argparse
import coremltools
import numpy as np
from keras.models import load_model

parser = argparse.ArgumentParser(description='Converts Keras .h5 model to CoreML .mlmodel')
parser.add_argument('--model', dest='model_name', help='Input model name')
parser.add_argument('--desc', dest='desc', help='Short description')
parser.add_argument('--input_desc', dest='input_desc', help='Input description')
parser.add_argument('--output_desc', dest='output_desc', help='Oouput description')
parser.add_argument('--author', dest='author', help='Author')
parser.add_argument('--license', dest='license', help='License')

args = parser.parse_args()

def main():
	model_name = args.model_name
	keras_model = load_model(model_name)

	coreml_model = coremltools.converters.keras.convert(keras_model, \
		input_names='input', output_names='output')

	coreml_model.input_description['input'] = args.input_desc
	coreml_model.output_description['output'] = args.output_desc

	coreml_model.short_description = args.desc
	coreml_model.author = args.author
	coreml_model.license = args.license

	f_name, f_ext = os.path.splitext(model_name)
	coreml_model.save(os.path.join(f_name + '.mlmodel'))

	print("Success!")

if __name__ == '__main__':
	main()	