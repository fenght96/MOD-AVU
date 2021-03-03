import os.path as osp
import tempfile

import mmcv
import numpy as np

from .custom import CustomDataset
from .builder import DATASETS
import xml.etree.ElementTree as ET


@DATASETS.register_module()
class LGDRgbtDataset(CustomDataset):

    CLASSES = ('bus', 'bicycle', 'car', 'person', 'truck', 'tricycle')

    def __init__(self, min_size=None, **kwargs):
        super(LGDRgbtDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i  for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size

    def load_annotations(self, ann_file):
        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        # both modalities
        for img_id in img_ids:
            filename = [f'rgb/{img_id}.jpg', 
                        f'trm/{img_id}.jpg']
            xml_path = osp.join(self.img_prefix, 'xml',
                                '{}.xml'.format(img_id))
            width = 640
            height = 512
            data_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        return data_infos

    def get_ann_info(self, idx):
        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'xml',
                            '{}.xml'.format(img_id))
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                #print(img_id, name)
                continue
            label = self.cat2label[name]
            difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)

        '''
        with open(txt_path, 'r') as f:
            objs = f.readlines()
        for obj in objs:
            obj_split = obj.split(' ')
            if True:
                name = obj_split[0].strip()
                label = self.cat2label[name]
                difficult = 0
                bbox = [
                    int(obj_split[1]),
                    int(obj_split[2]),
                    int(obj_split[1]) + int(obj_split[3]),
                    int(obj_split[2]) + int(obj_split[4])
                ]
                ignore = False
                if self.min_size:
                    assert not self.test_mode
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if w < self.min_size or h < self.min_size:
                        ignore = True
                if difficult or ignore:
                    bboxes_ignore.append(bbox)
                    labels_ignore.append(label)
                else:
                    bboxes.append(bbox)
                    labels.append(label)
            else:
                pass
        '''
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann

    def format_results(self, results, txtfile_prefix=None, **kwargs):
        """Format the results to txt for KAIST evaluation.

        Args:
            results (list): Testing results of the dataset.
            txtfile_prefix (str | None): The prefix of txt files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the txt filepaths, tmp_dir is the temporal directory created
                for saving txt files when txtfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if txtfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            txtfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # write results into txt file
        with open(txtfile_prefix + '.txt', 'w') as f:
            # i-th image
            for i in range(len(results)):
                # j-th detection result of 'person' class
                for j in range(np.shape(results[i][0])[0]):
                     box = results[i][0][j,:].tolist()
                     box[2] = box[2] - box[0]
                     box[3] = box[3] - box[1]
                     f.write(str(i+1)+',')
                     f.write(','.join(str(c) for c in box))
                     f.write('\n')
