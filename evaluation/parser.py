import lxml.etree
import os
import glob
import numpy as np

def get_truth(file):
    tree = lxml.etree.parse(file)
    root = tree.getroot()
    truth = root[2]

    frames = dict()
    for frame in truth.iter('frame'):
        index = int(frame.attrib['index'])
        points = dict()
        for point in frame.iter('point'):
            name = point.attrib['name']
            x = float(point.attrib['x'])
            y = float(point.attrib['y'])
            points[name] = (x,y)
        frames[index] = points
    return frames


def create_box(frame):
    box = np.ndarray(shape = (4,2), dtype = np.int64)
    tl = frame.get('tl')
    tr = frame.get('tr')
    br = frame.get('br')
    bl = frame.get('bl')
    box[0] = tl[0], tl[1]
    box[1] = tr[0], tr[1]
    box[2] = br[0], br[1]
    box[3] = bl[0], bl[1]
    return box
    

if __name__ == '__main__':
    file = "tax003.gt.xml"
    frames = get_truth(file)
    print(frames.get(200).get('bl')[0])
    print(frames.get(1))
    create_box(frames.get(1))





