import json
import ndjson
import os
import os.path as osp
from glob import glob
import cv2
from tqdm import tqdm
import math


class CutOut2FitBBox:
    def __init__(self, data_path: str, out_dir: str = './output/', *, token_list: list = []) -> None:
        self._dataset = osp.dirname(data_path)  # nuscenes_ped, nuimages_ped, waymo_ped
        self._version = osp.basename(data_path)  # v1.0-train, ...
        self._img_dir = data_path + '/img/'
        self._json_dir = data_path + '/json/'
        self._out_dir = out_dir
        self._bbox_info = []
        self.token_list = token_list
        self.warning_list = []

        os.makedirs(self._out_dir, exist_ok=True)
        os.makedirs('./warning', exist_ok=True)

    def _add_bbox_info(self, ped_token: str, bbox: list, dataset: str, version: str, img_token: str):
        self._bbox_info.append({
            'ped_token': ped_token,
            'bbox': bbox,
            'dataset': dataset,
            'version': version,
            'img_token': img_token
        })

    def read_all_frame(self):
        _frames = glob(self._img_dir + '*')
        for _frame in tqdm(_frames, desc=self._dataset+'('+self._version+')'):
            _frame_name = osp.splitext(osp.basename(_frame))[0]
            _bbox_per_frame = self._read_json(_frame_name)
            self._cut_img(_frame, _bbox_per_frame)

    def _cut_img(self, _frame: str, _bbox_dict: dict):
        _ori_img = cv2.imread(_frame)
        _ori_size = _ori_img.shape
        for _token, _bbox in _bbox_dict.items():
            self.token_list.append(_token)
            _cutted_img = _ori_img[
                _bbox[1]: _bbox[3],  # top : bottom
                _bbox[0]: _bbox[2]   # left: right
            ]
            if _bbox[0] < 1 or _bbox[1] < 1 or _bbox[2] > _ori_size[1]-1 or _bbox[3] > _ori_size[0]-1:
                cv2.imwrite("./warning/" + _token + ".jpg", _cutted_img)
                self.warning_list.append({"token": _token, "img": _frame})
                continue
            cv2.imwrite(self._out_dir + _token + ".jpg", _cutted_img)

    def _read_json(self, _frame_name: str) -> dict:
        _dst = {}
        _json_path = self._json_dir + _frame_name + '.json'
        with open(_json_path, 'r') as f:
            _ndj = ndjson.load(f)
            _tmp = json.dumps(_ndj)
            _ndj = json.loads(_tmp)
        for _record in _ndj:
            _bbox = _record['bbox']
            _bbox = [
                math.floor(_bbox[0]), math.floor(_bbox[1]),
                math.ceil(_bbox[2]), math.ceil(_bbox[3])
            ]
            _ped_token = _record['token']
            _cnt = 0
            while _ped_token in self.token_list:
                _cnt += 1
                _ped_token = _ped_token + '_' + str(_cnt)
            _dst[_ped_token] = _bbox
            self._add_bbox_info(_ped_token, _bbox, self._dataset, self._version, _frame_name)
        return _dst

    def export_json(self):
        _json_path = self._out_dir + 'ped_info.json'
        with open(_json_path, 'a') as f:
            writer = ndjson.writer(f)
            for d in self._bbox_info:
                writer.writerow(d)
        with open('./warning/warning.json', 'a') as f:
            writer = ndjson.writer(f)
            for d in self.warning_list:
                writer.writerow(d)


def main():
    datasets = glob('./img_ped/*')
    token_list = []
    for dataset_src in datasets:
        versions = glob(dataset_src + '/*')
        for version_src in tqdm(versions):
            co2b = CutOut2FitBBox(version_src, token_list=token_list)
            co2b.read_all_frame()
            token_list = co2b.token_list
            co2b.export_json()


if __name__ == "__main__":
    main()
