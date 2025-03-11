import torch
import torch.nn.functional as F


import os
import math
import json
#import tqdm
import numpy as np
from simple_knn._C import distCUDA2
from sh_utils import RGB2SH
import seaborn as sns
import cv2

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)

def draw_humansd_skeleton(image, pose, mmpose_detection_thresh=0.3, height=None, width=None, humansd_skeleton_width=10):
    humansd_skeleton = [
        [1, 0, 1],
        [0, 0, 2],
        [3, 1, 3],
        [2, 2, 4],
        [5, 3, 5],
        [4, 4, 6],
        [7, 5, 7],
        [6, 6, 8],
        [9, 7, 9],
        [8, 8, 10],
        [11, 5, 11],
        [10, 6, 12],
        [13, 11, 13],
        [12, 12, 14],
        [15, 13, 15],
        [14, 14, 16],
    ]
    # humansd_skeleton_width=10
    humansd_color = sns.color_palette("hls", len(humansd_skeleton))

    def plot_kpts(img_draw, kpts, color, edgs, width):
        for idx, kpta, kptb in edgs:
            if kpts[kpta, 2] > mmpose_detection_thresh and \
                    kpts[kptb, 2] > mmpose_detection_thresh:
                line_color = tuple([int(255 * color_i) for color_i in color[idx]])

                cv2.line(img_draw, (int(kpts[kpta, 0]), int(kpts[kpta, 1])), (int(kpts[kptb, 0]), int(kpts[kptb, 1])),
                         line_color, width)
                cv2.circle(img_draw, (int(kpts[kpta, 0]), int(kpts[kpta, 1])), width // 2, line_color, -1)
                cv2.circle(img_draw, (int(kpts[kptb, 0]), int(kpts[kptb, 1])), width // 2, line_color, -1)

    if image is None:
        pose_image = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        pose_image = np.array(image, dtype=np.uint8)
    for person_i in range(len(pose)):
        if np.sum(pose[person_i]) > 0:
            plot_kpts(pose_image, pose[person_i], humansd_color, humansd_skeleton, humansd_skeleton_width)

    return pose_image

PRESET = {
    '2head': {"nose": [0.0021412528585642576, 0.09663597494363785, 0.0658300444483757], "neck": [-0.0010681250132620335, -0.00773019902408123, 0.0070612248964607716], "right_shoulder": [-0.05952326953411102, -0.01074729859828949, -9.221061918651685e-05], "right_elbow": [-0.08530594408512115, -0.055266208946704865, -0.0002990829525515437], "right_wrist": [-0.10009504109621048, -0.10672871768474579, 0.04145783558487892], "left_shoulder": [0.05843103677034378, -0.00984001811593771, 0.006993727292865515], "left_elbow": [0.08632118254899979, -0.05415782332420349, 0.0014727215748280287], "left_wrist": [0.10457248985767365, -0.10312359780073166, 0.03887445852160454], "right_hip": [0.045034412294626236, -0.11504126340150833, 0.013827108778059483], "right_knee": [0.04918656870722771, -0.18811877071857452, 0.013533061370253563], "right_ankle": [0.04868278279900551, -0.26335108280181885, 0.00865345261991024], "left_hip": [-0.042053237557411194, -0.12201181799173355, 0.01380667183548212], "left_knee": [-0.04329617694020271, -0.1887422651052475, 0.017976703122258186], "left_ankle": [-0.043604593724012375, -0.2664816379547119, 0.009848599322140217], "right_eye": [-0.059466104954481125, 0.14732250571250916, 0.03459283709526062], "left_eye": [0.05259571596980095, 0.15527287125587463, 0.03957919031381607], "right_ear": [-0.0915362536907196, 0.12822526693344116, -0.0068031493574380875], "left_ear": [0.0850512906908989, 0.132415309548378, -0.0023928845766931772]},
    '2.5head': {"nose": [0.0017373580485582352, 0.10973469167947769, 0.0656559020280838], "neck": [0.0005578577402047813, 0.04446818679571152, 0.007351499516516924], "right_shoulder": [-0.06537579745054245, 0.04295105114579201, 0.0007656214293092489], "right_elbow": [-0.10073791444301605, -0.011567563749849796, 0.0011918626260012388], "right_wrist": [-0.13308240473270416, -0.08192941546440125, 0.041919875890016556], "left_shoulder": [0.06524424999952316, 0.044958289712667465, 0.006919004488736391], "left_elbow": [0.10230650007724762, -0.003759381826967001, 0.0006817418616265059], "left_wrist": [0.13201594352722168, -0.0703246220946312, 0.037093378603458405], "right_hip": [0.04595021530985832, -0.08364222943782806, 0.014006344601511955], "right_knee": [0.04918656870722771, -0.18811877071857452, 0.013533061370253563], "right_ankle": [0.04868278279900551, -0.26335108280181885, 0.00865345261991024], "left_hip": [-0.04423205181956291, -0.09601262211799622, 0.014173348434269428], "left_knee": [-0.04329617694020271, -0.1887422651052475, 0.017976703122258186], "left_ankle": [-0.043604593724012375, -0.2664816379547119, 0.009848599322140217], "right_eye": [-0.059466104954481125, 0.14732250571250916, 0.03459283709526062], "left_eye": [0.05259571596980095, 0.15527287125587463, 0.03957919031381607], "right_ear": [-0.0915362536907196, 0.12822526693344116, -0.0068031493574380875], "left_ear": [0.0850495845079422, 0.1380147635936737, -0.002471053972840309]},
    '3head': {"nose": [-0.009920584969222546, 0.12076142430305481, 0.05144921690225601], "neck": [-0.010807228274643421, 0.0514158271253109, 0.0013299279380589724], "right_shoulder": [-0.06006723269820213, 0.04696957394480705, 0.0002115728275384754], "right_elbow": [-0.09811610728502274, -0.005623028613626957, 0.0059844981878995895], "right_wrist": [-0.13300932943820953, -0.06305573135614395, 0.03299811854958534], "left_shoulder": [0.03998754173517227, 0.04971584677696228, 0.00508818868547678], "left_elbow": [0.08185750991106033, -0.0020600110292434692, -0.00042760284850373864], "left_wrist": [0.12347455322742462, -0.057821620255708694, 0.03114679642021656], "right_hip": [0.028485914692282677, -0.07230513542890549, 0.007468733470886946], "right_knee": [0.02956254966557026, -0.1690925806760788, 0.0041032107546925545], "right_ankle": [0.03433872014284134, -0.26075273752212524, 0.004083261359483004], "left_hip": [-0.045689668506383896, -0.07934730499982834, 0.007511031813919544], "left_knee": [-0.04399966821074486, -0.17574959993362427, 0.004589484538882971], "left_ankle": [-0.04372630640864372, -0.26631468534469604, 0.004584244918078184], "right_eye": [-0.050951384007930756, 0.14704833924770355, 0.030185498297214508], "left_eye": [0.030040200799703598, 0.14831678569316864, 0.03128870204091072], "right_ear": [-0.07488956302404404, 0.12157893925905228, -0.004280052147805691], "left_ear": [0.05265972018241882, 0.12078605592250824, -0.004687455017119646]},
    '4head': {"nose": [-0.003130262019112706, 0.16587696969509125, 0.05414091795682907], "neck": [-0.008572826161980629, 0.10935179889202118, -0.005226037930697203], "right_shoulder": [-0.0681774765253067, 0.10397181659936905, -0.006579247768968344], "right_elbow": [-0.11421658098697662, 0.04033476859331131, 0.0004059926141053438], "right_wrist": [-0.1564374417066574, -0.02915881760418415, 0.033092476427555084], "left_shoulder": [0.05288884416222572, 0.10729481279850006, -0.0006785409059375525], "left_elbow": [0.10355149209499359, 0.0446460098028183, -0.007352650165557861], "left_wrist": [0.1539081186056137, -0.022825559601187706, 0.030852381139993668], "right_hip": [0.03897187486290932, -0.040350597351789474, 0.0022019187454134226], "right_knee": [0.04027460888028145, -0.15746350586414337, -0.001870364649221301], "right_ankle": [0.04605376720428467, -0.2683720886707306, -0.001894504064694047], "left_hip": [-0.05078059807419777, -0.04887162148952484, 0.002253100508823991], "left_knee": [-0.04873568192124367, -0.16551849246025085, -0.0012819726252928376], "left_ankle": [-0.04840493202209473, -0.27510207891464233, -0.0012883121380582452], "right_eye": [-0.03098677098751068, 0.19395537674427032, 0.019874906167387962], "left_eye": [0.01657041721045971, 0.1956009715795517, 0.02724142000079155], "right_ear": [-0.05411602929234505, 0.1733667254447937, -0.013280442915856838], "left_ear": [0.0373358279466629, 0.16922003030776978, -0.009465649724006653]},
    '7head': {"nose": [0.008811305277049541, 0.31194087862968445, 0.03809100389480591], "neck": [0.002824489725753665, 0.2497633546590805, -0.027212638407945633], "right_shoulder": [-0.06274063885211945, 0.2438453733921051, -0.0287011731415987], "right_elbow": [-0.11721517890691757, 0.11645109206438065, -0.020040860399603844], "right_wrist": [-0.14608919620513916, -0.027798010036349297, 0.013604634441435337], "left_shoulder": [0.07043224573135376, 0.24750067293643951, -0.02221038192510605], "left_elbow": [0.13446204364299774, 0.11769299954175949, -0.02934686467051506], "left_wrist": [0.1729350984096527, -0.029831381514668465, 0.013683688826858997], "right_hip": [0.05391363054513931, -0.017539208754897118, -0.0190418753772974], "right_knee": [0.06667664647102356, -0.1525234431028366, -0.023521387949585915], "right_ankle": [0.07457379996776581, -0.3374432921409607, -0.02354794181883335], "left_hip": [-0.059334054589271545, -0.023282308131456375, -0.018985575065016747], "left_knee": [-0.06742465496063232, -0.14818398654460907, -0.02287415601313114], "left_ankle": [-0.08158088475465775, -0.33846616744995117, -0.02288113348186016], "right_eye": [-0.0218308437615633, 0.34282705187797546, 0.0003984148206654936], "left_eye": [0.03048212267458439, 0.3446371853351593, 0.008501569740474224], "right_ear": [-0.04727301374077797, 0.3201795518398285, -0.0360724963247776], "left_ear": [0.05332399904727936, 0.31561821699142456, -0.03187622129917145]},
    '8head': {"nose": [0.013792905025184155, 0.3043023347854614, 0.031688809394836426], "neck": [0.009342477656900883, 0.2587159276008606, -0.022332727909088135], "right_shoulder": [-0.05664093792438507, 0.24332502484321594, -0.02351135015487671], "right_elbow": [-0.09274981170892715, 0.12393242120742798, -0.014478711411356926], "right_wrist": [-0.12370569258928299, -0.0065268343314528465, 0.014483341947197914], "left_shoulder": [0.08111944049596786, 0.24344637989997864, -0.0181470587849617], "left_elbow": [0.11824966222047806, 0.12197338044643402, -0.021892601624131203], "left_wrist": [0.14754801988601685, -0.0040277112275362015, 0.014088761992752552], "right_hip": [0.05466757342219353, -0.027295198291540146, -0.015528455376625061], "right_knee": [0.07225559651851654, -0.18235255777835846, -0.018520904704928398], "right_ankle": [0.089942067861557, -0.3677787184715271, -0.019252480939030647], "left_hip": [-0.042525578290224075, -0.03484155982732773, -0.015481928363442421], "left_knee": [-0.06011202931404114, -0.180166095495224, -0.018695630133152008], "left_ankle": [-0.07281138002872467, -0.36362409591674805, -0.018701398745179176], "right_eye": [-0.011531195603311062, 0.329828143119812, 0.0005379004869610071], "left_eye": [0.0317026749253273, 0.33132410049438477, 0.007234722841531038], "right_ear": [-0.03255778178572655, 0.311111181974411, -0.02960335463285446], "left_ear": [0.05058026313781738, 0.30734145641326904, -0.026135355234146118]},
    '17point': {"nose": [0.013792905025184155, 0.3043023347854614, 0.031688809394836426], "left_eye": [0.0317026749253273, 0.33132410049438477, 0.007234722841531038], "right_eye": [-0.011531195603311062, 0.329828143119812, 0.0005379004869610071], "left_ear": [0.05058026313781738, 0.30734145641326904, -0.026135355234146118], "right_ear": [-0.03255778178572655, 0.311111181974411, -0.02960335463285446], "left_shoulder": [0.08111944049596786, 0.24344637989997864, -0.0181470587849617], "right_shoulder": [-0.05664093792438507, 0.24332502484321594, -0.02351135015487671], "left_elbow": [0.11824966222047806, 0.12197338044643402, -0.021892601624131203], "right_elbow": [-0.09274981170892715, 0.12393242120742798, -0.014478711411356926], "left_wrist": [0.14754801988601685, -0.0040277112275362015, 0.014088761992752552], "right_wrist": [-0.12370569258928299, -0.0065268343314528465, 0.014483341947197914], "left_hip": [0.05466757342219353, -0.027295198291540146, -0.015528455376625061], "right_hip": [-0.042525578290224075, -0.03484155982732773, -0.015481928363442421], "left_knee": [0.07225559651851654, -0.18235255777835846, -0.018520904704928398], "right_knee": [-0.06011202931404114, -0.180166095495224, -0.018695630133152008], "left_ankle": [0.089942067861557, -0.3677787184715271, -0.019252480939030647], "right_ankle": [-0.07281138002872467, -0.36362409591674805, -0.018701398745179176]},
}



def joint_mapper_smplx_to_openpose18(joints):
    indices = (
        np.array(
            [
                56,  # nose
                13,  # neck
                18,  # right_shoulder
                20,  # right_elbow
                22,  # right_wrist
                17,  # left_shoulder
                19,  # left_elbow
                21,  # left_wrist
                3,  # right_hip
                6,  # right_knee
                9,  # right_ankle
                2,  # left_hip
                5,  # left_knee
                8,  # left_ankle
                57,  # right_eye
                58,  # left_eye
                59,  # right_ear
                60,  # left_ear
            ],
            dtype=np.int64,
        )
        - 1
    )
    return joints[indices]

def joint_mapper_smplx_to_humansd17(joints):
    indices = np.array([
        56, # nose
        58, # left_eye
        57, # right_eye
        60, # left_ear
        59, # right_ear
        17, # left_shoulder
        18, # right_shoulder
        19, # left_elbow
        20, # right_elbow
        21, # left_wrist
        22, # right_wrist
        2, # left_hip
        3, # right_hip
        5, # left_knee
        6, # right_knee
        8, # left_ankle
        9, # right_ankle
    ], dtype=np.int64) - 1
    return joints[indices]

class Skeleton:
    def __init__(self, motion_path, humansd_style=True):
        # init pose [18, 3], in [-1, 1]^3
        self.style = "humansd" if humansd_style else "openpose"
#        self.apose = apose
        if self.style == "humansd":
            # init pose [17, 3], in [-1, 1]^3
            self.points3D = np.array([
                [-0.00313026,  0.16587697,  0.05414092],
                [ 0.01657042,  0.19560097,  0.02724142],
                [-0.03098677,  0.19395538,  0.01987491],
                [ 0.03733583,  0.16922003, -0.00946565],
                [-0.05411603,  0.17336673, -0.01328044],
                [ 0.05288884,  0.10729481, -0.00067854],
                [-0.06817748,  0.10397182, -0.00657925],
                [ 0.10355149,  0.04464601, -0.00735265],
                [-0.11421658,  0.04033477,  0.00040599],
                [ 0.15390812, -0.02282556,  0.03085238],
                [-0.15643744, -0.02915882,  0.03309248],
                [-0.0507806 , -0.04887162,  0.0022531 ],
                [ 0.03897187, -0.0403506 ,  0.00220192],
                [-0.04873568, -0.16551849, -0.00128197],
                [ 0.04027461, -0.15746351, -0.00187036],
                [-0.04840493, -0.27510208, -0.00128831],
                [ 0.04605377, -0.26837209, -0.0018945 ],
            ], dtype=np.float32)

            self.name = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
            # lines [16, 2]
            self.lines = np.array([
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 4],
                [3, 5],
                [4, 6],
                [5, 7],
                [6, 8],
                [7, 9],
                [8, 10],
                [5, 11],
                [6, 12],
                [11, 13],
                [12, 14],
                [13, 15],
                [14, 16]
            ], dtype=np.int32)
        else:
            # init pose [18, 3], in [-1, 1]^3
            self.points3D = np.array([
                [-0.00313026,  0.16587697,  0.05414092],
                [-0.00857283,  0.1093518 , -0.00522604],
                [-0.06817748,  0.10397182, -0.00657925],
                [-0.11421658,  0.04033477,  0.00040599],
                [-0.15643744, -0.02915882,  0.03309248],
                [ 0.05288884,  0.10729481, -0.00067854],
                [ 0.10355149,  0.04464601, -0.00735265],
                [ 0.15390812, -0.02282556,  0.03085238],
                [ 0.03897187, -0.0403506 ,  0.00220192],
                [ 0.04027461, -0.15746351, -0.00187036],
                [ 0.04605377, -0.26837209, -0.0018945 ],
                [-0.0507806 , -0.04887162,  0.0022531 ],
                [-0.04873568, -0.16551849, -0.00128197],
                [-0.04840493, -0.27510208, -0.00128831],
                [-0.03098677,  0.19395538,  0.01987491],
                [ 0.01657042,  0.19560097,  0.02724142],
                [-0.05411603,  0.17336673, -0.01328044],
                [ 0.03733583,  0.16922003, -0.00946565]
            ], dtype=np.float32)

            self.name = ["nose", "neck", "right_shoulder", "right_elbow", "right_wrist", "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye", "left_eye", "right_ear", "left_ear"]
            # lines [17, 2]
            self.lines = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [0, 14], [14, 16], [0, 15], [15, 17]], dtype=np.int32)

        # keypoint color [18, 3]
        # color as in controlnet_aux (https://github.com/patrickvonplaten/controlnet_aux/blob/master/src/controlnet_aux/open_pose/util.py#L94C5-L96C73)
        self.colors = [
            [255, 0, 0],
            [255, 85, 0],
            [255, 170, 0],
            [255, 255, 0],
            [170, 255, 0],
            [85, 255, 0],
            [0, 255, 0],
            [0, 255, 85],
            [0, 255, 170],
            [0, 255, 255],
            [0, 170, 255],
            [0, 85, 255],
            [0, 0, 255],
            [85, 0, 255],
            [170, 0, 255],
            [255, 0, 255],
            [255, 0, 170],
            [255, 0, 85],
        ]

        # smplx mesh if available
        self.smplx_model = None
        self.vertices = None
        self.faces = None
        self.ori_center = None
        self.ori_scale = None

        self.body_pose = np.zeros((21, 3), dtype=np.float32)
        self.body_orient = np.zeros(3, dtype=np.float32)
        
        self.body_pose_tpose = torch.Tensor(self.body_pose)
        # let's default to A-pose
        # self.body_pose[15, 2] = -0.7853982
        # self.body_pose[16, 2] = 0.7853982
        # self.body_pose[0, 1] = 0.2
        # self.body_pose[0, 2] = 0.1
        # self.body_pose[1, 1] = -0.2
        # self.body_pose[1, 2] = -0.1
        """ SMPLX body_pose definition
        0: 'left_hip',#'L_Hip', XYZ -> (-X)(-Y)Z, 后外高 -> 前里高 (3) XYZ
        1: 'right_hip',#'R_Hip', (4) XYZ -> (-X)(-Y)Z, 后里低 -> 前外低 (4) XYZ
        2: 'spine1',#'Spine1', (-X)Y(-Z) -> (0) XYZ
        3: 'left_knee',#'L_Knee', 同左UpperLeg
        4: 'right_knee',#'R_Knee',同右UpperLeg
        5: 'spine2',
        6: 'left_ankle',
        7: 'right_ankle',#'R_Ankle',同右UpperLeg
        8: 'spine3',#'Spine3', (-X)Y(-Z) 同脊椎
        9: 'left_foot',#'L_Foot',同左UpperLeg
        10: 'right_foot',#'R_Foot',同右UpperLeg
        11: 'neck',#'Neck', (-X)Y(-Z) 同脊椎
        12: 'left_collar',#'L_Collar', XYZ -> ZXY (VRM), 前拧, 后, 高 -> 高, 前拧, 后 (1) YZX
        13: 'right_collar',#'R_Collar', XYZ -> (-Z)(-X)Y , 前拧, 前, 低 -> 高, 后拧, 前 (2) YZX
        14: 'head',#'Head', (-X)Y(-Z) 同脊椎
        15: 'left_shoulder',#'L_Shoulder', 同左肩膀
        16: 'right_shoulder',#'R_Shoulder', 同右肩膀
        17: 'left_elbow',#'L_Elbow', 同左肩膀
        18: 'right_elbow',#'R_Elbow', 同右肩膀
        19: 'left_wrist',#'L_Wrist', 同左肩膀
        20: 'right_wrist',#'R_Wrist', 同右肩膀
        """

        self.left_hand_pose = np.zeros((15, 3), dtype=np.float32)
        self.right_hand_pose = np.zeros((15, 3), dtype=np.float32)
        """ hand_pose definition
        index, middle, pinky, ring, thumb; each with 3 joints.
        """

        # gaussian model
#        self.gs = Renderer(sh_degree=0, white_background=False)
#        self.gs.gaussians.load_ply(opt.ply)

        # motion data
#        self.motion_seq = np.load(opt.motion)["poses"][:, 1:22]
#        self.seq_id = time_step
        self.seq_id = -1
        
        motions = np.load(motion_path)
        self.num_frames = motions["poses"].shape[0]
        
        self.motion_seq = motions["pose_body"]
        self.motion_seq = self.motion_seq.reshape(self.num_frames, 21, 3)
        
        self.left_hand_pose_seq = motions['pose_hand'][:, :45]
        self.left_hand_pose_seq = self.left_hand_pose_seq.reshape(self.num_frames, 15, 3)
        
        self.right_hand_pose_seq = motions['pose_hand'][:, 45:]
        self.right_hand_pose_seq = self.right_hand_pose_seq.reshape(self.num_frames, 15, 3)
        
        # self.Rh_seq = np.load(motion_path)["Rh"]
        self.Th_seq = motions["trans"]
        
        self.betas = torch.Tensor(motions["betas"]).unsqueeze(0)
        
        # gaussian center to smplx faces mapping
        self.mapping_dist = None
        self.mapping_face = None
        self.mapping_uvw = None
        
    def load_smplx(self, path, time_step=-1, betas=None, expression=None, gender="neutral", blender=False):
        import smplx
        
        self.seq_id = time_step
        if self.seq_id >= 0:
            self.body_pose = np.array(self.motion_seq[self.seq_id % self.num_frames])
            self.left_hand_pose = np.array(self.left_hand_pose_seq[self.seq_id % self.num_frames])
            self.right_hand_pose = np.array(self.right_hand_pose_seq[self.seq_id % self.num_frames])
            # self.body_orient = np.array(self.Rh_seq[self.seq_id % len(self.motion_seq)])
            # betas = self.betas
        # else:
            betas = None
        if self.smplx_model is None:
            self.smplx_model = smplx.create(
                path,
                model_type="smplx",
                gender=gender,
                use_face_contour=False,
                num_betas=16,
                num_expression_coeffs=10,
                ext="npz",
                use_pca=False,  # explicitly control hand pose
                flat_hand_mean=True,  # use a flatten hand default pose
            )
        # betas = torch.randn([1, self.smplx_model.num_betas], dtype=torch.float32)
        # expression = torch.randn([1, self.smplx_model.num_expression_coeffs], dtype=torch.float32)

        smplx_output = self.smplx_model(
#            global_orient=torch.tensor(-self.body_orient, dtype=torch.float32).unsqueeze(0),
            body_pose=torch.tensor(self.body_pose, dtype=torch.float32).unsqueeze(0),
            left_hand_pose=torch.tensor(
                self.left_hand_pose, dtype=torch.float32
            ).unsqueeze(0),
            right_hand_pose=torch.tensor(
                self.right_hand_pose, dtype=torch.float32
            ).unsqueeze(0),
            betas=betas,
            # betas=self.betas,
            expression=expression,
            return_verts=True,
            return_full_pose=True,
        )
        
        # smplx_output_tpose = smplx_model(
        #     body_pose=torch.tensor(self.body_pose_tpose, dtype=torch.float32).unsqueeze(0),
        #     # betas=torch.tensor(default_pose['shape'], dtype=torch.float32).unsqueeze(0),
        #     betas=betas, 
        #     expression=expression, 
        #     return_verts=True
        # )

        self.vertices = smplx_output.vertices.detach().cpu().numpy()[0]  # [10475, 3]
        self.faces = torch.Tensor(np.asarray(self.smplx_model.faces).astype(np.int64)).to(torch.int64)  # [20908, 3]

        # self.vertices_tpose = smplx_output_tpose.vertices.detach().cpu().numpy()[0]
        # self.faces_tpose = self.smplx_model.faces
        
        # tmp: save deformed smplx mesh
        # import trimesh
        # _mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        # _mesh.export('smplx.obj')
        
        # joints_tpose = smplx_output_tpose.joints.detach().cpu().numpy()[0]  # [127, 3]
        # if self.style == "humansd":
        #     joints_tpose = joint_mapper_smplx_to_humansd17(joints_tpose)
        # else:
        #     joints_tpose = joint_mapper_smplx_to_openpose18(joints_tpose)

        # self.points3D_tpose = np.concatenate(
        #     [joints_tpose, np.ones_like(joints_tpose[:, :1])], axis=1
        # )  # [18, 4] or [17, 4]
        # if blender:
        #     # coordinate system: opengl --> blender (switch y/z)
        #     self.points3D_tpose[:, [1, 2]] = self.points3D_tpose[:, [2, 1]]

        # # rescale and recenter
        # if self.ori_center_tpose is None:
        #     vmin_tpose = self.vertices_tpose.min(0)
        #     vmax_tpose = self.vertices_tpose.max(0)
        #     self.ori_center_tpose = (vmax_tpose + vmin_tpose) / 2
        #     self.ori_scale_tpose = 0.6 / np.max(vmax_tpose - vmin_tpose)

        # self.vertices_tpose = (self.vertices_tpose - self.ori_center_tpose) * self.ori_scale_tpose
        # self.points3D_tpose[:, :3] = (self.points3D_tpose[:, :3] - self.ori_center_tpose) * self.ori_scale_tpose
        self.params = {}
        self.params['R'] = torch.Tensor(np.eye(3).astype(np.float32)).cuda()
        self.params['Th'] = torch.Tensor(np.zeros((1,3)).astype(np.float32)).cuda()
        self.params['shapes'] = torch.Tensor(np.zeros((1,10)).astype(np.float32)).cuda()
        # self.params['poses'] = torch.Tensor(np.zeros((1,72)).astype(np.float32)).cuda()
        self.params['poses'] = torch.Tensor(smplx_output.full_pose)
        # body_pose = self.body_pose.reshape(1, 63)
        # self.params['poses'][:, :-6] = torch.Tensor(body_pose[:, :]).cuda()
        self.params['betas'] = torch.Tensor(self.betas).cuda()
        self.params['faces'] = torch.Tensor(np.asarray(self.faces).astype(np.int64)).to(torch.int64).cuda()
        self.params['shapedirs'] = torch.Tensor(self.smplx_model.shapedirs).cuda()
        self.params['posedirs'] = torch.Tensor(self.smplx_model.posedirs).cuda()
        self.params['J_regressor'] = torch.Tensor(self.smplx_model.J_regressor).cuda()
        self.params['parents'] = torch.Tensor(self.smplx_model.parents).cuda()
        self.params['lbs_weights'] = torch.Tensor(self.smplx_model.lbs_weights).cuda()
        
        joints = smplx_output.joints.detach().cpu().numpy()[0]  # [127, 3]
        if self.style == "humansd":
            joints = joint_mapper_smplx_to_humansd17(joints)
        else:
            joints = joint_mapper_smplx_to_openpose18(joints)

        self.points3D = np.concatenate(
            [joints, np.ones_like(joints[:, :1])], axis=1
        )  # [18, 4] or [17, 4]
        if blender:
            # coordinate system: opengl --> blender (switch y/z)
            self.points3D[:, [1, 2]] = self.points3D[:, [2, 1]]

        # rescale and recenter
        if self.ori_center is None:
            vmin = self.vertices.min(0)
            vmax = self.vertices.max(0)
            self.ori_center = (vmax + vmin) / 2
            self.ori_scale = 0.6 / np.max(vmax - vmin)

        self.vertices = (self.vertices - self.ori_center) * self.ori_scale
        self.points3D[:, :3] = (self.points3D[:, :3] - self.ori_center) * self.ori_scale

        self.scale(-10)  # rescale

        colors =  np.ones_like(self.vertices) * 0.5
        self.vertices = torch.tensor(np.asarray(self.vertices)).float().cuda()
        self.colors = RGB2SH(torch.tensor(np.asarray(colors)).float()).cuda()

        dist2 = torch.clamp_min(distCUDA2(self.vertices), 0.0000001)
        self.scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        self.rots = torch.zeros((self.vertices.shape[0], 4)).cuda()
        self.rots[:, 0] = 1

        self.opacities = inverse_sigmoid(0.1 * torch.ones((self.vertices.shape[0], 1), dtype=torch.float)).cuda()
        
    def scale(self, delta):
        self.points3D[:, :3] *= 1.1 ** (-delta)
        if self.vertices is not None:
            self.vertices *= 1.1 ** (-delta)
        # if self.vertices_tpose is not None:
        #     self.points3D_tpose[:, :3] *= 1.1 ** (-delta)
        #     self.vertices_tpose *= 1.1 ** (-delta)
            
    def sample_smplx_points(self, N=20000, time_step=-1):
        # if has smplx mesh, sample from the surface
        assert self.vertices is not None
        import trimesh
        
        mesh = trimesh.Trimesh(self.vertices.cpu().numpy(), self.faces)
        # mesh.export(f'check_pcd_{time_step}.obj')
        samples, _ = trimesh.sample.sample_surface(mesh, N)
        return samples
            
    def draw(self, mvp, H, W, enable_occlusion=False):
        # mvp: [4, 4]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        points = self.points3D @ mvp.T # [18, 4]
        points = points[:, :3] / points[:, 3:] # NDC in [-1, 1]

        xs = (points[:, 0] + 1) / 2 * H # [18]
        ys = (points[:, 1] + 1) / 2 * W # [18]
        mask = (xs >= 0) & (xs < H) & (ys >= 0) & (ys < W)

        # hide certain keypoints based on empirical occlusion
        if enable_occlusion:
            # decide view by the position of nose between two ears
            if points[0, 2] > points[-1, 2] and points[0, 2] < points[-2, 2]:
                # left view
                mask[-2] = False # no right ear
                if xs[-4] > xs[-3]:
                    mask[-4] = False # no right eye if it's "righter" than left eye
            elif points[0, 2] < points[-1, 2] and points[0, 2] > points[-2, 2]:
                # right view
                mask[-1] = False
                if xs[-3] < xs[-4]:
                    mask[-3] = False
            elif points[0, 2] > points[-1, 2] and points[0, 2] > points[-2, 2]:
                # back view
                mask[0] = False # no nose
                mask[-3] = False # no eyes
                mask[-4] = False

        # 18 points
        for i in range(18):
            if not mask[i]: continue
            cv2.circle(canvas, (int(xs[i]), int(ys[i])), 4, self.colors[i], thickness=-1)

        # 17 lines
        for i in range(17):
            cur_canvas = canvas.copy()
            if not mask[self.lines[i]].all(): 
                continue
            X = xs[self.lines[i]]
            Y = ys[self.lines[i]]
            mY = np.mean(Y)
            mX = np.mean(X)
            length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))
            polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), 4), int(angle), 0, 360, 1)
            
            cv2.fillConvexPoly(cur_canvas, polygon, self.colors[i])
            
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        
        canvas = canvas.astype(np.float32) / 255
        return canvas, np.stack([xs, ys], axis=1)
    
    def humansd_draw(self, mvp, H, W, enable_occlusion=False):
        # mvp: [4, 4]
        # canvas = np.zeros((H, W, 3), dtype=np.uint8)

        points = self.points3D @ mvp.T # [17, 4]
        points = points[:, :3] / points[:, 3:] # NDC in [-1, 1]

        xs = (points[:, 0] + 1) / 2 * H # [17]
        ys = (points[:, 1] + 1) / 2 * W # [17]
        kp_coord = np.stack([xs, ys], axis=1) # [17, 2]
        kp_conf = np.array([1.] * 17, dtype=np.float32)

        if enable_occlusion:
            # 0 - nose, 1 - left eye, 2 - right eye, 3 - left ear, 4 - right ear
            # decide view by the position of nose between two ears
            if points[0, 2] > points[3, 2] and points[0, 2] < points[4, 2]:
                # left view
                kp_conf[4] = 0. # no right ear
                if xs[2] > xs[1]:
                    kp_conf[2] = 0. # no right eye if it's "righter" than left eye
            elif points[0, 2] < points[3, 2] and points[0, 2] > points[4, 2]:
                # right view
                kp_conf[3] = 0.
                if xs[1] < xs[2]:
                    kp_conf[1] = 0.
            elif points[0, 2] > points[3, 2] and points[0, 2] > points[4, 2]:
                # back view
                kp_conf[0] = 0. # no nose
                kp_conf[1] = 0. # no eyes
                kp_conf[2] = 0.

        kp = np.concatenate([kp_coord[np.newaxis, ...], kp_conf[np.newaxis, ..., np.newaxis]], axis=-1)

        whole_draw = draw_humansd_skeleton(
            image=None,
            pose=kp,
            height=H,
            width=W,
            humansd_skeleton_width=int(10 * H / 512),
        )
        # pose_img = Image.fromarray(whole_draw)
        # control_image = torch.from_numpy(whole_draw).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0
        control_image = whole_draw.astype(np.float32) / 255.0
        return control_image, kp