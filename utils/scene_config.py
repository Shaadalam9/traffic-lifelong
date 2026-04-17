from __future__ import annotations

import xml.etree.ElementTree as ET

from tqdm.auto import tqdm

from utils.base import PipelineStage
from utils.io_utils import write_json


class SceneConfigBuilder(PipelineStage):
    def run(self) -> None:
        if not self.config.annotations_xml.exists():
            raise FileNotFoundError(f"Annotations XML not found: {self.config.annotations_xml}")

        tree = ET.parse(self.config.annotations_xml)
        root = tree.getroot()

        image_name = self.config.scene_image_name
        image_node = None
        image_nodes = root.findall('.//image')
        self.logger.info(f'Found {len(image_nodes)} image node(s) in annotations XML')

        for node in tqdm(image_nodes, desc='Scanning annotation images', unit='image'):
            if image_name is None or node.get('name') == image_name:
                image_node = node
                break

        if image_node is None:
            raise ValueError('Could not find a matching <image> entry in annotations XML')

        output = {
            'image_name': image_node.get('name'),
            'width': int(float(image_node.get('width', '0'))),
            'height': int(float(image_node.get('height', '0'))),
            'regions': {},
        }

        children = list(image_node)
        for child in tqdm(children, desc='Exporting scene regions', unit='region'):
            label = child.get('label')
            if not label:
                continue
            if child.tag in {'polyline', 'polygon'}:
                points = []
                for pair in child.get('points', '').split(';'):
                    if not pair.strip():
                        continue
                    x_str, y_str = pair.split(',')
                    points.append([
                        round(float(x_str), self.config.scene_round_digits),
                        round(float(y_str), self.config.scene_round_digits),
                    ])
                output['regions'][label] = {
                    'type': child.tag,
                    'points': points,
                }
            elif child.tag == 'box':
                xtl = round(float(child.get('xtl', '0')), self.config.scene_round_digits)
                ytl = round(float(child.get('ytl', '0')), self.config.scene_round_digits)
                xbr = round(float(child.get('xbr', '0')), self.config.scene_round_digits)
                ybr = round(float(child.get('ybr', '0')), self.config.scene_round_digits)
                output['regions'][label] = {
                    'type': 'box',
                    'points': [[xtl, ytl], [xbr, ybr]],
                }

        write_json(self.config.scene_regions_json, output)
        self.logger.info(
            f"Scene config written to {self.config.scene_regions_json} "
            f"with {len(output['regions'])} region(s)"
        )
