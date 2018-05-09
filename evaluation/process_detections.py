from lxml import etree 
from pathlib import Path
from io import StringIO
import os
import IOU
import pandas as pd

'''
Metrics
    Per-frame:
        IOU
    Per-clip:
        Average IOU
        # missed detections
    Per-document type:
        Average IOU
        # missed detections
    Per-background:
        Average IOU
        # missed detections
'''

def write_results(results_df, filename = 'detection_results.csv'):
    results_df.to_csv(path_or_buf = filename, header=True, index='document')

def process_all_results(test_path, results_path, backgrounds):
    overall_results = pd.DataFrame(columns=['background', 'document', 'frame', 'IOU', 'speed (secs)', 'rejected'])
    for background in backgrounds:
        gt_folder = test_path + background + '/groundtruth/'
        results_folder = results_path + background + '/'
        results_df = process_background_results(gt_folder=gt_folder, results_folder=results_folder, background=background)
        overall_results.append(results_df)
    return overall_results

def process_background_results(gt_folder, results_folder, background):
    results_df = pd.DataFrame(columns=['background', 'document', 'frame', 'IOU', 'speed (secs)', 'rejected'])
    for filename in os.listdir(path = results_folder):
        if filename.endswith(".xml"):
            result_df = process_clip_results(gt_folder = gt_folder, result_path = results_folder, result_fname = filename, background = background)
        else:
            print("Encountered non-XML file in path " + filename)
            continue
        results_df = results_df.append(result_df, ignore_index = True)
        print(results_df)
    return results_df

def process_clip_results(gt_folder, result_path, result_fname, background):
        '''
        gt_folder: FULL PATH to the groundtruth folder for a particular background (e.g., ../testDataset/background01/groundtruth/)
        result_path: FULL PATH to a result folder for a particular background (e.g., ../results/test/background01/)
        result_fname: JUST the filename of the result XML file (e.g., letter003.xml)
        background: JUST the name of the background this clip corresponds to (e.g., background01). In reality, could just pull this off the end of gt_folder.
        '''
        result_df = pd.DataFrame(columns=['background', 'document', 'frame', 'IOU', 'speed (secs)', 'rejected'])
        document = result_fname.split('.')[0]
        gt_file = gt_folder + document + '.gt.xml'
        if Path(gt_file).exists():
            gt_frames = get_xml_frames(xml_file = gt_file)
            result_frames = get_xml_frames(xml_file = result_path + result_fname)
            for i, gt_frame in enumerate(gt_frames):
                result_frame = result_frames[i]
                res_index, res_rejected = get_frame_info(xml_frame=result_frame)
                gt_index, _ = get_frame_info(xml_frame = gt_frame)
                speed = speed_from_xml_frame(xml_frame = result_frame)
                assert res_index == gt_index 
                if res_rejected:
                    result_df.loc[i] = [background, document, res_index, -1, speed, 1]
                    continue
                '''Form box tuples (tl, tr, br, bl) from XML groundtruth and XML results'''
                gt_box = box_from_xml_frame(xml_frame = gt_frame)
                res_box = box_from_xml_frame(xml_frame = result_frame)
                iou = IOU.IOU(gt_corners=gt_box, detected_corners = res_box)
                # print("IOU for frame " + str(i) + " of " + document + ": " + str(iou))
                result_df.loc[i] = [background, document, res_index, iou, speed, 0]
        else:
            print("Couldn't find groundtruth file corresponding to " + gt_file + " in process_background_results()")
        return result_df

def get_xml_frames(xml_file):
    root = etree.parse(xml_file).getroot()
    seg_results = root.find('segmentation_results')
    frames = seg_results.findall('frame')
    return frames

def get_frame_info(xml_frame):
    frame_number = xml_frame.get('index')
    rejected = xml_frame.get('rejected')
    reject = False
    if rejected == 'true':
        print(rejected)
        reject = True
    return frame_number, reject
    
def box_from_xml_frame(xml_frame):
    points = xml_frame.findall('point')
    bl, tl, tr, br = points[0], points[1], points[2], points[3]
    bl = (round(int(float(bl.get('x')))), round(int(float(bl.get('y')))))
    tl = (round(int(float(tl.get('x')))), round(int(float(tl.get('y')))))
    tr = (round(int(float(tr.get('x')))), round(int(float(tr.get('y')))))
    br = (round(int(float(br.get('x')))), round(int(float(br.get('y')))))
    box = (bl, tl, tr, br)
    return box

def speed_from_xml_frame(xml_frame):
    speed = xml_frame.find('speed')
    seconds = speed.get('secs')
    return seconds


if __name__ == '__main__':

    '''DATA PATHS'''
    data_root = '/Users/echristiansen/Data/SmartDoc/challenge_1_seg/'

    sample_root = 'sampleDataset/'
    sample_data = data_root + sample_root + 'input_sample/background00/'
    sample_gt_path = data_root + sample_root + 'input_sample_groundtruth/background00_gt/'

    sample_files = ['datasheet001.avi', 'letter001.avi', 'magazine001.avi']
    sample_gts = ['datasheet001.gt.xml', 'letter001.gt.xml', 'magazine001.gt.xml']

    '''TEST DATA PATHS'''
    test_data = data_root + 'testDataset/'
    backgrounds = ['background01', 'background02', 'background03', 'background04', 'background05']

    '''RESULTS PATHS'''
    results_path = data_root + 'results/'
    sample_results_path = results_path + 'sample/'
    test_results_path = results_path + 'test/'

    # '''Process overall results'''
    # overall_results = process_all_results(test_path = test_data, results_path = test_results_path, backgrounds = backgrounds)
    # write_results(overall_results)

    '''Process background results'''
    background = 'background05'
    gt_path = test_data + background + '/groundtruth/'
    result_path = test_results_path + background + '/'
    background_results = process_background_results(gt_folder=gt_path, results_folder=result_path, background=background)
    write_results(results_df = background_results, filename=background+'_results.csv')

    '''Did this and it worked'''
    # gt_path = test_data + 'background01/groundtruth/'
    # ## result_path = sample_results_path + 'background00/'
    # result_path = test_results_path + 'background01/'
    # result_fname = 'letter003.xml'
    # ## results_df = process_clip_results(gt_folder = sample_gt_path, result_path = result_path, result_fname = result_fname, background='background00')
    # results_df = process_clip_results(gt_folder = gt_path, result_path = result_path, result_fname = result_fname, background='background01')
    # print(results_df)
    # write_results(results_df)



    


     








