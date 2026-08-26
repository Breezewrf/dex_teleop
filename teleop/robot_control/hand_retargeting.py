from dex_retargeting import RetargetingConfig
from pathlib import Path
import yaml
from enum import Enum
import logging_mp
logger_mp = logging_mp.getLogger(__name__)

class HandType(Enum):
    INSPIRE_HAND = "./assets/inspire_hand/inspire_hand.yml"
    INSPIRE_HAND_Unit_Test = "./assets/inspire_hand/inspire_hand.yml"
    UNITREE_DEX3 = "./assets/unitree_hand/unitree_dex3.yml"
    UNITREE_DEX3_Unit_Test = "./assets/unitree_hand/unitree_dex3.yml"
    BRAINCO_HAND = "./assets/brainco_hand/brainco.yml"
    BRAINCO_HAND_Unit_Test = "./assets/brainco_hand/brainco.yml"
    CASIA_HAND = "./assets/casia_hand_M/casia.yml"
    CASIA_HAND_Unit_Test = "./assets/casia_hand_M/casia.yml"
    OMNIHAND = "./assets/o12_hand_description-o12_t3/omnihand.yml"
    OMNIHAND_Unit_Test = "./assets/o12_hand_description-o12_t3/omnihand.yml"
    OMNIHAND_VECTOR = "./assets/o12_hand_description-o12_t3/omnihand_vector.yml"
    OMNIHAND_VECTOR_Unit_Test = "./assets/o12_hand_description-o12_t3/omnihand_vector.yml"

class HandRetargeting:
    def __init__(self, hand_type: HandType):
        if hand_type == HandType.UNITREE_DEX3:
            RetargetingConfig.set_default_urdf_dir('./assets')
        elif hand_type == HandType.UNITREE_DEX3_Unit_Test:
            RetargetingConfig.set_default_urdf_dir('./assets')
        elif hand_type == HandType.INSPIRE_HAND:
            RetargetingConfig.set_default_urdf_dir('./assets')
        elif hand_type == HandType.INSPIRE_HAND_Unit_Test:
            RetargetingConfig.set_default_urdf_dir('./assets')
        elif hand_type == HandType.BRAINCO_HAND:
            RetargetingConfig.set_default_urdf_dir('./assets')
        elif hand_type == HandType.BRAINCO_HAND_Unit_Test:
            RetargetingConfig.set_default_urdf_dir('./assets')
        elif hand_type == HandType.CASIA_HAND:
            RetargetingConfig.set_default_urdf_dir('./assets')
        elif hand_type == HandType.CASIA_HAND_Unit_Test:
            RetargetingConfig.set_default_urdf_dir('./assets')
        elif hand_type == HandType.OMNIHAND:
            RetargetingConfig.set_default_urdf_dir('./assets')
        elif hand_type == HandType.OMNIHAND_Unit_Test:
            RetargetingConfig.set_default_urdf_dir('./assets')
        elif hand_type == HandType.OMNIHAND_VECTOR:
            RetargetingConfig.set_default_urdf_dir('./assets')
        elif hand_type == HandType.OMNIHAND_VECTOR_Unit_Test:
            RetargetingConfig.set_default_urdf_dir('./assets')

        config_file_path = Path(hand_type.value)

        try:
            with config_file_path.open('r') as f:
                self.cfg = yaml.safe_load(f)
                
            if 'left' not in self.cfg or 'right' not in self.cfg:
                raise ValueError("Configuration file must contain 'left' and 'right' keys.")

            left_retargeting_config = RetargetingConfig.from_dict(self.cfg['left'])
            right_retargeting_config = RetargetingConfig.from_dict(self.cfg['right'])
            self.left_retargeting = left_retargeting_config.build()
            self.right_retargeting = right_retargeting_config.build()

            self.left_retargeting_joint_names = self.left_retargeting.joint_names
            self.right_retargeting_joint_names = self.right_retargeting.joint_names
            self.left_indices = self.left_retargeting.optimizer.target_link_human_indices
            self.right_indices = self.right_retargeting.optimizer.target_link_human_indices

            if hand_type == HandType.UNITREE_DEX3 or hand_type == HandType.UNITREE_DEX3_Unit_Test:
                # In section "Sort by message structure" of https://support.unitree.com/home/en/G1_developer/dexterous_hand
                self.left_dex3_api_joint_names  = [ 'left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint',
                                                    'left_hand_middle_0_joint', 'left_hand_middle_1_joint', 
                                                    'left_hand_index_0_joint', 'left_hand_index_1_joint' ]
                self.right_dex3_api_joint_names = [ 'right_hand_thumb_0_joint', 'right_hand_thumb_1_joint', 'right_hand_thumb_2_joint',
                                                    'right_hand_middle_0_joint', 'right_hand_middle_1_joint',
                                                    'right_hand_index_0_joint', 'right_hand_index_1_joint' ]
                self.left_dex_retargeting_to_hardware = [ self.left_retargeting_joint_names.index(name) for name in self.left_dex3_api_joint_names]
                self.right_dex_retargeting_to_hardware = [ self.right_retargeting_joint_names.index(name) for name in self.right_dex3_api_joint_names]

            elif hand_type == HandType.INSPIRE_HAND or hand_type == HandType.INSPIRE_HAND_Unit_Test:
                # "Joint Motor Sequence" of https://support.unitree.com/home/en/G1_developer/inspire_dfx_dexterous_hand
                self.left_inspire_api_joint_names  = [ 'L_pinky_proximal_joint', 'L_ring_proximal_joint', 'L_middle_proximal_joint',
                                                       'L_index_proximal_joint', 'L_thumb_proximal_pitch_joint', 'L_thumb_proximal_yaw_joint' ]
                self.right_inspire_api_joint_names = [ 'R_pinky_proximal_joint', 'R_ring_proximal_joint', 'R_middle_proximal_joint',
                                                       'R_index_proximal_joint', 'R_thumb_proximal_pitch_joint', 'R_thumb_proximal_yaw_joint' ]
                self.left_dex_retargeting_to_hardware = [ self.left_retargeting_joint_names.index(name) for name in self.left_inspire_api_joint_names]
                self.right_dex_retargeting_to_hardware = [ self.right_retargeting_joint_names.index(name) for name in self.right_inspire_api_joint_names]
            
            elif hand_type == HandType.BRAINCO_HAND or hand_type == HandType.BRAINCO_HAND_Unit_Test:
                # "Driver Motor ID" of https://www.brainco-hz.com/docs/revolimb-hand/product/parameters.html
                self.left_brainco_api_joint_names  = [ 'left_thumb_metacarpal_joint', 'left_thumb_proximal_joint', 'left_index_proximal_joint',
                                                       'left_middle_proximal_joint', 'left_ring_proximal_joint', 'left_pinky_proximal_joint' ]
                self.right_brainco_api_joint_names = [ 'right_thumb_metacarpal_joint', 'right_thumb_proximal_joint', 'right_index_proximal_joint',
                                                       'right_middle_proximal_joint', 'right_ring_proximal_joint', 'right_pinky_proximal_joint' ]
                self.left_dex_retargeting_to_hardware = [ self.left_retargeting_joint_names.index(name) for name in self.left_brainco_api_joint_names]
                self.right_dex_retargeting_to_hardware = [ self.right_retargeting_joint_names.index(name) for name in self.right_brainco_api_joint_names]
            elif hand_type == HandType.CASIA_HAND or hand_type == HandType.CASIA_HAND_Unit_Test:
                # Sim2Sim
                # self.left_casia_api_joint_names  = ['index1', 'index2', 'index3', 'little1', 'little2', 'little3', 'middle1', 'middle2', 'middle3', 'ring1', 'ring2', 'ring3', 'thumb1', 'thumb2']
                # self.right_casia_api_joint_names = ['index1', 'index2', 'index3', 'little1', 'little2', 'little3', 'middle1', 'middle2', 'middle3', 'ring1', 'ring2', 'ring3', 'thumb1', 'thumb2']

                self.left_casia_api_joint_names  = ['left_index_proximal', 'left_index_intermediate', 'left_index_distal',
                       'left_pinky_proximal', 'left_pinky_intermediate', 'left_pinky_distal',
                       'left_middle_proximal', 'left_middle_intermediate', 'left_middle_distal',
                       'left_ring_proximal', 'left_ring_intermediate', 'left_ring_distal',
                       'left_thumb_proximal', 'left_thumb_intermediate']
                self.right_casia_api_joint_names = ['right_index_proximal', 'right_index_intermediate', 'right_index_distal',
                       'right_pinky_proximal', 'right_pinky_intermediate', 'right_pinky_distal',
                       'right_middle_proximal', 'right_middle_intermediate', 'right_middle_distal',
                       'right_ring_proximal', 'right_ring_intermediate', 'right_ring_distal',
                       'right_thumb_proximal', 'right_thumb_intermediate']

                # Sim2Real, Do not uncomment following lines, the sim2real retarget has been done in robot_to_real_mapping in Casia_Controller.
                # self.left_casia_api_joint_names  = ['thumb1', 'thumb2', 'index1', 'middle1', 'ring1', 'little1','index2', 'middle2', 'ring2', 'little2']
                # self.right_casia_api_joint_names = ['thumb1', 'thumb2', 'index1', 'middle1', 'ring1', 'little1','index2', 'middle2', 'ring2', 'little2']
                self.left_dex_retargeting_to_hardware = [ self.left_retargeting_joint_names.index(name) for name in self.left_casia_api_joint_names]
                self.right_dex_retargeting_to_hardware = [ self.right_retargeting_joint_names.index(name) for name in self.right_casia_api_joint_names]
            elif hand_type in (
                HandType.OMNIHAND,
                HandType.OMNIHAND_Unit_Test,
                HandType.OMNIHAND_VECTOR,
                HandType.OMNIHAND_VECTOR_Unit_Test,
            ):
                # Only the 12 active joints are commanded. The remaining seven joints
                # are coupled through URDF mimic tags for retargeting and MJCF equality
                # constraints in MuJoCo.
                self.left_omnihand_api_joint_names = [
                    'L_thumb_roll_joint', 'L_thumb_abad_joint',
                    'L_thumb_mcp_joint', 'L_thumb_pip_joint',
                    'L_index_abad_joint', 'L_index_mcp_joint', 'L_index_pip_joint',
                    'L_middle_abad_joint', 'L_middle_mcp_joint', 'L_middle_pip_joint',
                    'L_ring_mcp_joint', 'L_pinky_mcp_joint',
                ]
                self.right_omnihand_api_joint_names = [
                    'R_thumb_roll_joint', 'R_thumb_abad_joint',
                    'R_thumb_mcp_joint', 'R_thumb_pip_joint',
                    'R_index_abad_joint', 'R_index_mcp_joint', 'R_index_pip_joint',
                    'R_middle_abad_joint', 'R_middle_mcp_joint', 'R_middle_pip_joint',
                    'R_ring_mcp_joint', 'R_pinky_mcp_joint',
                ]
                self.left_dex_retargeting_to_hardware = [
                    self.left_retargeting_joint_names.index(name)
                    for name in self.left_omnihand_api_joint_names
                ]
                self.right_dex_retargeting_to_hardware = [
                    self.right_retargeting_joint_names.index(name)
                    for name in self.right_omnihand_api_joint_names
                ]
        except FileNotFoundError:
            logger_mp.warning(f"Configuration file not found: {config_file_path}")
            raise
        except yaml.YAMLError as e:
            logger_mp.warning(f"YAML error while reading {config_file_path}: {e}")
            raise
        except Exception as e:
            logger_mp.error(f"An error occurred: {e}")
            raise
