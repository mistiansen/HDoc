import Levenshtein
import itertools
import os
import argparse


'''
Evaluation Process

So maybe there's two parts to this:
    1. Evaluates the detection and hough in terms of IOU
    2. Evaluates just Hough/perspective transform/orientation correction for OCR accuracy

OCR process: 
    1. Pull full OCR on each original and processed image
    2. Write OCR to text files named the same (corresponding to image name) but in different directories (OCR/processed, OCR/unprocessed)
    3. Iterate over text file pairs and compute Levenshtein distance of each from the ground truth 
'''

def compare_files(files):
    distances = dict()
    results = []
    for a, b in itertools.combinations(files, 2):
        print("comparing " + a + " with " + b)
        distance = Levenshtein.distance(get_text(a), get_text(b))
        print(distance)
        distances[a] = [b, distance]
    return distances

# def get_text(filename):
#     text_file = open(filename, 'r')
#     return text_file.read()

def get_text(filename):
    with open(filename, 'r') as data_file:
        # text=data_file.read().replace('\n', '')
        text=data_file.read()
    return text

def get_filenames(directory):
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            full_path = os.path.join(directory, filename)
            filenames.append(full_path)
    return filenames

def process_OCR(ocr_gt = '/Users/echristiansen/Data/SmartDoc/challenge_2_ocr/sampleDataset/input_sample_groundtruth/', ocr_unproc=None, ocr_proc=None):
    filename = ocr_gt + '03630.txt'
    text = get_text(filename)
    print(text)
    

if __name__ == '__main__':
    # filenames = get_filenames("pics/experiments/thesis/IADOT/")
    # for filename in filenames:
    #     print filename
    # distances = compare_files(filenames)
    # print distances

    process_OCR()
    
