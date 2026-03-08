#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import xml.etree.ElementTree as ET
from copy import deepcopy

def indent(elem, level=0):
    """递归缩进，兼容低版本 Python（等价于 Python 3.9+ 的 ET.indent）。"""
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            indent(e, level+1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

def make_collision_from_visual(visual_elem):
    """由 <visual> 生成 <collision>。仅复制 <origin> 与 <geometry>。"""
    collision = ET.Element('collision')
    # 复制 name（若有）
    vname = visual_elem.get('name')
    if vname:
        collision.set('name', vname)
    # 复制 <origin> 与 <geometry>
    for child in list(visual_elem):
        if child.tag in ('origin', 'geometry'):
            collision.append(deepcopy(child))
    return collision

def has_same_named_collision(link_elem, name):
    """在同一 <link> 下是否已有同名 <collision>。"""
    if not name:
        return False
    for ch in link_elem:
        if ch.tag == 'collision' and ch.get('name') == name:
            return True
    return False

def process_urdf(input_path, output_path=None):
    tree = ET.parse(input_path)
    root = tree.getroot()

    # 遍历所有 link
    for link in root.findall('.//link'):
        # 注意：需要基于快照 list(link) 来遍历并做插入，否则插入会影响迭代
        children = list(link)
        i = 0
        while i < len(children):
            elem = children[i]
            if elem.tag == 'visual':
                vname = elem.get('name')
                # 如果已经有同名 collision，跳过
                if has_same_named_collision(link, vname):
                    i += 1
                    continue
                # 构造 collision，并在 link 中紧跟 visual 后面插入
                collision = make_collision_from_visual(elem)
                # 计算当前 elem 在 link 中的真实索引（因为 link 可能已有插入/删除）
                real_children = list(link)
                try:
                    real_idx = real_children.index(elem)
                except ValueError:
                    # 理论上不会发生；保险处理
                    real_idx = len(real_children) - 1
                link.insert(real_idx + 1, collision)

                # 同步本地快照（保持 while 正确前进）
                children.insert(i + 1, collision)
                i += 2
            else:
                i += 1

    # 美化缩进并写文件
    indent(root)
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + '.collision' + (ext if ext else '.urdf')

    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f'Done. Wrote: {output_path}')

if __name__ == '__main__':
    input_path = 'right.urdf'
    output_path = 'right_col.urdf'
    process_urdf(input_path, output_path)

