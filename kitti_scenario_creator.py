import xml.etree.ElementTree as ET
import os
import math

MAX_V = 10
MAX_A = 3

class myObj(object):
    pass

def disRotate(radius):
    if radius < 0:
        radius = radius + 2 * math.pi
    comp = 2*math.pi - radius
    return math.floor(comp*6/math.pi)

def disVelocity(velocity):
    disV = math.floor(velocity/MAX_V*5)
    if disV >= 5:
        return 5
    if disV <= -5:
        return -5
    return disV

def disAcceleration(acceleration):
    disA = math.floor(acceleration/MAX_A*5)
    if disA >= 5:
        return 5
    if disA <= -5:
        return -5
    return disA

dirOxts = "oxts"
xmlTracklet = "tracklet_labels.xml"

lbObjs = ET.parse(xmlTracklet)
rootObjs = lbObjs.getroot()

tracklets = rootObjs[0]
countObjs = int(tracklets[0].text)

items = []
i = 0
while i < countObjs:
    objIndex = i + 2
    item = myObj()
    item.obj = tracklets[objIndex]
    item.type = tracklets[objIndex][0].text
    item.fstFrame = int(tracklets[objIndex][4].text)
    poses = tracklets[objIndex][5]
    item.cntFrames = int(poses[0].text)
    item.boxesInFrames = []
    j = 0
    while j < item.cntFrames:
        boxIndex = j + 2
        boxObj = poses[boxIndex]
        b = myObj()
        b.type = item.type
        b.frameIndex = j + item.fstFrame
        b.tx = float(boxObj[0].text)
        b.ty = float(boxObj[1].text)
        b.tz = float(boxObj[2].text)
        b.rx = float(boxObj[3].text)
        b.ry = float(boxObj[4].text)
        b.rz = float(boxObj[5].text)
        b.state = int(boxObj[6].text)
        b.occ = int(boxObj[7].text)
        b.tru = int(boxObj[8].text)
        item.boxesInFrames.append(b)
        j += 1
    items.append(item)
    i += 1

# boxes of all the frames
frameBoxes = {}

for item in items:
    for box in item.boxesInFrames:
        if not frameBoxes.get(box.frameIndex):
            frameBoxes[box.frameIndex] = []
        frameBoxes[box.frameIndex].append(box)

# for key in frames:
#     for b in frames[key]:
#         print(b.frameIndex, b.type, b.tx)

# egos off a frames
egos = {}

for root,dirs,files in os.walk(dirOxts):
    for file in files:
        frameIndex = int(file.replace(".txt",""))
        f = open(os.path.join(root,file),"r")
        strIn = f.read().split()
        ego = myObj()
        ego.vn = float(strIn[6])
        ego.ve = float(strIn[7])
        ego.vu = float(strIn[10])
        ego.ax = float(strIn[11])
        ego.ay = float(strIn[12])
        egos[frameIndex] = ego
        f.close()

# dumpxml
xml = open("result.xml","w")
scenarios = ET.Element("scenarios")

i = 0
while i < countObjs:
    # find closest left(y>0)
    # find closest right(y<0)
    # find closest 2(x-axis)
    left_box = None
    right_box = None
    closest_box_1 = None
    closest_box_2 = None
    if frameBoxes.get(i):
        for box in frameBoxes[i]:
            if box.type == "Misc":
                continue
            if box.ty < 0:
                if left_box is None or left_box.ty < box.ty:
                    left_box = box
            if box.ty >= 0:
                if right_box is None or right_box.ty > box.ty:
                    right_box = box
            if closest_box_1 is None or closest_box_1.tx > box.tx:
                closest_box_2 = closest_box_1
                closest_box_1 = box

    if left_box is None and right_box is None and closest_box_1 is None and closest_box_2 is None:
        i += 1
        continue

    scenario = ET.SubElement(scenarios, "scenario")
    scenario.attrib
    ego_vn = ET.SubElement(scenario,"ego_vn")
    ego_vn.text = str(disVelocity(egos[i].vn))
    ego_ve = ET.SubElement(scenario,"ego_ve")
    ego_ve.text = str(disVelocity(egos[i].ve))
    ego_ax = ET.SubElement(scenario,"ego_ax")
    ego_ax.text = str(disAcceleration(egos[i].ax))
    ego_ay = ET.SubElement(scenario,"ego_ay")
    ego_ay.text = str(disAcceleration(egos[i].ay))

    if left_box is None:
        left_box_existence = ET.SubElement(scenario,"left_box_existence")
        left_box_existence.text = str(False)
        left_box_rotation = ET.SubElement(scenario,"left_box_rotation")
        left_box_rotation.text = str(0)
        left_box_type = ET.SubElement(scenario,"left_box_type")
        left_box_type.text = str("Misc")
        left_box_occlusion = ET.SubElement(scenario,"left_box_occlusion")
        left_box_occlusion.text = str(box.occ)
    else:
        left_box_existence = ET.SubElement(scenario,"left_box_existence")
        left_box_existence.text = str(True)
        left_box_rotation = ET.SubElement(scenario,"left_box_rotation")
        left_box_rotation.text = str(disRotate(left_box.rz))
        left_box_type = ET.SubElement(scenario,"left_box_type")
        left_box_type.text = str(left_box.type)
        left_box_occlusion = ET.SubElement(scenario,"left_box_occlusion")
        left_box_occlusion.text = str(left_box.occ)

    if right_box is None:
        right_box_existence = ET.SubElement(scenario,"right_box_existence")
        right_box_existence.text = str(False)
        right_box_rotation = ET.SubElement(scenario,"right_box_rotation")
        right_box_rotation.text = str(0)
        right_box_type = ET.SubElement(scenario,"right_box_type")
        right_box_type.text = str("Misc")
        right_box_occlusion = ET.SubElement(scenario,"right_box_occlusion")
        right_box_occlusion.text = str(box.occ)
    else:
        right_box_existence = ET.SubElement(scenario,"right_box_existence")
        right_box_existence.text = str(True)
        right_box_rotation = ET.SubElement(scenario,"right_box_rotation")
        right_box_rotation.text = str(disRotate(right_box.rz))
        right_box_type = ET.SubElement(scenario,"right_box_type")
        right_box_type.text = str(right_box.type)
        right_box_occlusion = ET.SubElement(scenario,"right_box_occlusion")
        right_box_occlusion.text = str(right_box.occ)

    if closest_box_1 is None:
        closest_box_1_existence = ET.SubElement(scenario,"closest_box_1_existence")
        closest_box_1_existence.text = str(False)
        closest_box_1_rotation = ET.SubElement(scenario,"closest_box_1_rotation")
        closest_box_1_rotation.text = str(0)
        closest_box_1_type = ET.SubElement(scenario,"closest_box_1_type")
        closest_box_1_type.text = str("Misc")
        closest_box_1_occlusion = ET.SubElement(scenario,"closest_box_1_occlusion")
        closest_box_1_occlusion.text = str(box.occ)
    else:
        closest_box_1_existence = ET.SubElement(scenario,"closest_box_1_existence")
        closest_box_1_existence.text = str(True)
        closest_box_1_rotation = ET.SubElement(scenario,"closest_box_1_rotation")
        closest_box_1_rotation.text = str(disRotate(closest_box_1.rz))
        closest_box_1_type = ET.SubElement(scenario,"closest_box_1_type")
        closest_box_1_type.text = str(closest_box_1.type)
        closest_box_1_occlusion = ET.SubElement(scenario,"closest_box_1_occlusion")
        closest_box_1_occlusion.text = str(closest_box_1.occ)

    if closest_box_2 is None:
        closest_box_2_existence = ET.SubElement(scenario,"closest_box_2_existence")
        closest_box_2_existence.text = str(False)
        closest_box_2_rotation = ET.SubElement(scenario,"closest_box_2_rotation")
        closest_box_2_rotation.text = str(0)
        closest_box_2_type = ET.SubElement(scenario,"closest_box_2_type")
        closest_box_2_type.text = str("Misc")
        closest_box_2_occlusion = ET.SubElement(scenario,"closest_box_2_occlusion")
        closest_box_2_occlusion.text = str(box.occ)
    else:
        closest_box_2_existence = ET.SubElement(scenario,"closest_box_2_existence")
        closest_box_2_existence.text = str(True)
        closest_box_2_rotation = ET.SubElement(scenario,"closest_box_2_rotation")
        closest_box_2_rotation.text = str(disRotate(closest_box_2.rz))
        closest_box_2_type = ET.SubElement(scenario,"closest_box_2_type")
        closest_box_2_type.text = str(closest_box_2.type)
        closest_box_2_occlusion = ET.SubElement(scenario,"closest_box_2_occlusion")
        closest_box_2_occlusion.text = str(closest_box_2.occ)
    i += 1

tree = ET.ElementTree()
tree._setroot(scenarios)
tree.write("aaa.xml")
xml.close()