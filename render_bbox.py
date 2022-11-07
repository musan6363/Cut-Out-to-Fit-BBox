"""
JupyterNotebookからの実行を想定

画像tokenと歩行者token(list)を入力して歩行者にBBoxを付与する．
(Waymo-ped-extractのときに設計したものを改変)
"""

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import ndjson
import os.path as osp


class RenderBBox:
    '''
    Parameters
    - src_img : path(str)
    '''

    def __init__(self, src_img: str) -> None:
        self.src_img = src_img
        self.src_name = osp.splitext(osp.basename(self.src_img))[0]
        _dataset_root = osp.dirname(osp.dirname(src_img))
        self.src_json = _dataset_root + '/json/' + self.src_name + '.json'

    def render(self) -> None:
        self.fig, self.ax = plt.subplots()

        self._render_bbox()
        self._render_img()
        self.ax.set_title(f"{self.src_name}")
        plt.show()

    def export(self, save_path: str) -> None:
        self.fig.savefig(save_path, format="png")

    def _get_bbox_info(self) -> None:
        self._ndj = self._load_ndjson()
        self._ped_token_bbox = {}
        for record in self._ndj:
            self._ped_token_bbox[record['token']] = record['bbox']

    def _load_ndjson(self) -> dict:
        with open(self.src_json, 'r') as f:
            _ndj = ndjson.load(f)
            _tmp = json.dumps(_ndj)
            _ndj = json.loads(_tmp)
        return _ndj

    def _render_bbox(self) -> None:
        self._get_bbox_info()
        for _bbox in self._ped_token_bbox.values():
            self.ax.add_patch(patches.Rectangle(
                xy=(_bbox[0], _bbox[1]),
                width=_bbox[2]-_bbox[0],
                height=_bbox[3]-_bbox[1],
                linewidth=1,
                edgecolor='red',
                facecolor='none'))

    def _render_img(self) -> None:
        self.im = Image.open(self.src_img)
        self.ax.imshow(self.im)
