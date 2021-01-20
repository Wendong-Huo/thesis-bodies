import numpy as np
from common import seeds

def get_a_limb(parent_length):
    
    limb = {
        "pos": np.random.random()*parent_length,
        "length": np.random.normal(loc=0.3, scale=0.2, size=[3])*np.random.choice([-1,1], size=[3]),
        "thickness": np.random.normal(loc=0.05, scale=0.01),
        "axis": np.random.choice([0,1,2]),
        "range": sorted(np.random.random(size=[2])*180),
        "limbs": np.random.randint(0,4),
    }
    # print("get a limb: \n", limb)
    return limb

worldbody = {
    "torso": True,
    "pos": np.array([0,0,0.0]),
    "length": np.array([0,0,1]),
    "thickness": 0.06,
    "limbs": 4,
}
def axis(i):
    ret = ['0']*3
    ret[i] = '1'
    return ' '.join(ret)
    


def to_xml(worldbody, height):
    global joint_number, max_depth, max_limb, lowest_point
    max_depth -=1
    if max_depth<0 or max_limb<0:
        print("Hit the depth limit.")
        xml = ""
    else:
        print(f"height: {height} (lowest_point: {lowest_point})")
        if lowest_point>height+worldbody['length'][2]:
            lowest_point=height+worldbody['length'][2]
        max_limb -=1
        if isinstance(worldbody, str):
            b = eval(worldbody)
        else:
            b = worldbody
        bodyname = "torso" if "torso" in b else f"L{joint_number}_D{max_depth}"
        xml = f"<body name='{bodyname}' pos='{b['pos'][0]:.02f} {b['pos'][1]:.02f} {b['pos'][2]:.02f}'>"
        if "torso" in b:
            pass
        else:
            xml += f"<joint axis='{axis(b['axis'])}' name='{bodyname}_joint' pos='0 0 0' range='{b['range'][0]:.02f} {b['range'][1]:.02f}' type='hinge'/>"
            joint_number += 1
        xml += f"<geom fromto='0 0 0 {b['length'][0]:.02f} {b['length'][1]:.02f} {b['length'][2]:.02f}' name='{bodyname}_geom' size='{b['thickness']:.02f}' type='capsule'/>"
        xml += f"<inertial mass='0.3'/>"
        for limb in range(b["limbs"]):
            new_limb = get_a_limb(b['length'])
            print("\t"*(3-max_depth), new_limb['pos'][2], new_limb['length'][2])
            new_height = height + new_limb['pos'][2]
            xml_limb = to_xml(new_limb, new_height)
            xml += xml_limb
        xml += f"</body>"
    max_depth +=1
    return xml

def reset_global_variables():
    global max_depth, joint_number, max_limb, lowest_point
    joint_number = 0
    max_depth = 3
    max_limb = 6
    lowest_point = 0

for i in range(100):
    reset_global_variables()
    with seeds.temp_seed(i):
        xml_worldbody = to_xml(worldbody, 0)
    print(f"lowest_point is {lowest_point}")
    adjusted_worldbody = worldbody.copy()
    above_ground = 0.1
    adjusted_worldbody["pos"][2] = -lowest_point + adjusted_worldbody["length"][2]/2 + above_ground
    print(f"\nadjusted to {adjusted_worldbody['pos'][2]}.\n")
    reset_global_variables()
    with seeds.temp_seed(i):
        xml_worldbody = to_xml(adjusted_worldbody, adjusted_worldbody["pos"][2])

    xml_template =f"""
                    <!--
                    joint_number: {joint_number}
                    -->
                    <mujoco model="randomrobot">
                        <compiler angle="degree" coordinate="global" inertiafromgeom="true"/>
                        <default>
                        <joint armature="0.01" damping=".1" limited="true"/>
                        <geom conaffinity="0" condim="3" contype="1" density="1000" friction="0.8 .1 .1" rgba="0.8 0.6 .4 1"/>
                        </default>
                        <option integrator="RK4" timestep="0.002"/>
                        <worldbody>
                    {xml_worldbody}
                        </worldbody>
                    </mujoco>
                    """
    with open(f"output_data/bodies/{100+i}.xml", "w") as f:
        print(xml_template, file=f)
