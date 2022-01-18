
import numpy as np
import numpy as np
import os
import pydot
import sys

import importlib
from scipy.spatial.transform import Rotation as R

import copy

if '/opt/drake/lib/python3.8/site-packages' not in sys.path:
    sys.path.append('/opt/drake/lib/python3.8/site-packages')



try:

    import pydrake
    from pydrake.all import (
        Adder, AddMultibodyPlantSceneGraph, Demultiplexer, DiagramBuilder,
        InverseDynamicsController, MakeMultibodyStateToWsgStateSystem,
        MeshcatVisualizerCpp, MultibodyPlant, Parser, PassThrough, FindResourceOrThrow,
        SchunkWsgPositionController, StateInterpolatorWithDiscreteDerivative,
        Simulator
    )

    from pydrake.all import (
        DiagramBuilder, MeshcatVisualizerCpp, MeshcatVisualizerParams,
        Simulator, FindResourceOrThrow,
        Parser, MultibodyPlant, RigidTransform, RollPitchYaw,
        PiecewisePolynomial, PiecewiseQuaternionSlerp, RotationMatrix, Solve,
        TrajectorySource, ConstantVectorSource
    )

    from pydrake.multibody import inverse_kinematics
    from pydrake.all import Parser

    from pydrake.all import (
        Adder, AddMultibodyPlantSceneGraph, Demultiplexer, DiagramBuilder,
        InverseDynamicsController, MakeMultibodyStateToWsgStateSystem,
        MeshcatVisualizerCpp, MultibodyPlant, Parser, PassThrough, FindResourceOrThrow,
        SchunkWsgPositionController, StateInterpolatorWithDiscreteDerivative,
        Simulator,
        DiagramBuilder, MeshcatVisualizerCpp, MeshcatVisualizerParams,
        Simulator, FindResourceOrThrow,
        Parser, MultibodyPlant, RigidTransform, RollPitchYaw,
        PiecewisePolynomial, PiecewiseQuaternionSlerp, RotationMatrix, Solve,
        TrajectorySource, ConstantVectorSource
    )
    from pydrake.multibody import inverse_kinematics
    from pydrake.all import MultibodyPlant
except:
    print("Could not import drake")
    #raise NotImplementedError


def FindResource(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def CreatePandaControllerPlant(dual=False, **kwargs):
    sys.path.append('/opt/drake/lib/python3.8/site-packages')

    """creates plant that includes only the robot and gripper, used for controllers."""
    sim_timestep = 1e-3
    plant_robot = MultibodyPlant(sim_timestep)
    if(dual):
        AddDualPanda(plant_robot, **kwargs)
    else:
        AddPanda(plant_robot, **kwargs)

    plant_robot.mutable_gravity_field().set_gravity_vector([0, 0, 0])
    plant_robot.Finalize()

    return plant_robot


def get_robot_qpos_from_obs(obs, n_arms = 1):
    """
    index 0-5: finger_pos
    index 6-11: fingers_vel
    index 12-24: len(qpos) = 13 = 4 + 7 + 2
        ['root_x_axis_joint', 'root_y_axis_joint', 'root_z_rotation_joint', 'linear_actuator_height',
        'right_panda_joint1', 'right_panda_joint2', 'right_panda_joint3', 'right_panda_joint4', 'right_panda_joint5', 'right_panda_joint6', 'right_panda_joint7',
        'right_panda_finger_joint1', 'right_panda_finger_joint2']
    index 25-37: qvel
    """
    if isinstance(obs, dict):
        agent_state = obs['agent']
    elif isinstance(obs, np.ndarray):
        len_agent_state = (4 + n_arms * 9) * 2 + n_arms * 12
        agent_state = obs[-len_agent_state:]
    else:
        raise NotImplementedError()
    s = agent_state
    s = s[n_arms * 12:]  # remove fingers_pos and fingers_vel
    qpos_mobile_base = s[:3]
    s = s[6:]  # remove base pos and vel
    s = s[:(1 + 9 * n_arms)]  # remove qvel
    s = np.concatenate([qpos_mobile_base, s])
    return s

    # qpos = agent_state[12:25]
    # qpos_gripper = agent_state[23:25]
    # return qpos, qpos_gripper

static_gripper = [0, 0]  ## qvel[-2:]
open_gripper = [1, 1]  ## qvel[-2:]
closed_gripper = [-1, -1]  ## qvel[-2:]
count = 0
last_q = []



def AddDualPanda(plant, scene_graph=None, ARM_ONLY=True, verbose=True, DRAKE=True, q0=None):
    """ rewrote AddIiwa() in https://github.com/RussTedrake/manipulation/blob/master/manipulation/scenarios.py
        combine with AddJointActuator() examples in AddPlanarGripper() in manipulation/forces.ipynb
    """


    model_file = FindResource("./models/robot/sciurus-for-drake/A2.urdf")
    if DRAKE:
        model_file = model_file.replace('single', 'drake')
#     print(model_file)
        
    if scene_graph == None:
        panda_model = Parser(plant).AddModelFromFile(model_file)
    else:
        panda_model = Parser(plant, scene_graph).AddModelFromFile(model_file)

    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("root"))       

    return panda_model

def AddPanda(plant, scene_graph=None, ARM_ONLY=True, verbose=True, DRAKE=True, q0=None):
    """ rewrote AddIiwa() in https://github.com/RussTedrake/manipulation/blob/master/manipulation/scenarios.py
        combine with AddJointActuator() examples in AddPlanarGripper() in manipulation/forces.ipynb
    """

    model_file = FindResource("./models/robot/sciurus-for-drake/A2_single.urdf")
    if ARM_ONLY:
        model_file = FindResource("./models/robot/sciurus-for-drake/A2_single_7links.urdf")
    if DRAKE:
        model_file = model_file.replace('single', 'drake')
#     print(model_file)
        
    if scene_graph == None:
        panda_model = Parser(plant).AddModelFromFile(model_file)
    else:
        panda_model = Parser(plant, scene_graph).AddModelFromFile(model_file)

    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("root"))
    
    """ without the above line there will be error:
    
            RuntimeError: DiagramBuilder::Connect: Mismatched vector sizes while connecting 
            output port iiwa7_continuous_state of System plant (size 27) to input port u0 of 
            System drake/systems/Demultiplexer@000055f8a5390400 (size 28)
            
        with a different link name there will be error:
            
            RuntimeError: This mobilizer is creating a closed loop since the outboard body 
            already has an inboard mobilizer connected to it. If a physical loop is really needed, 
            consider using a constraint instead.
    """

    actuated_joints = ['root_x_axis_joint', 'root_y_axis_joint', 'root_z_rotation_joint', 'linear_actuator_height',
                        'right_panda_joint1', 'right_panda_joint2', 'right_panda_joint3', 'right_panda_joint4',
                        'right_panda_joint5', 'right_panda_joint6', 'right_panda_joint7',
                        'right_panda_finger_joint1', 'right_panda_finger_joint2']
    if q0 == None:
        q0 = [0] * 7
    if not ARM_ONLY and len(q0) == 7:
        q0 = [0, 0, 0, 0] + q0 + [0, 0]
    
    index = 0
    for joint_name in actuated_joints:
        
        joint = plant.GetJointByName(joint_name)
        if joint.type_name() == 'prismatic':  ## gripper
#             joint.set_default_translation(0)
            joint.set_default_translation(q0[index])
            index += 1
        elif joint.type_name() == 'weld':
            continue
        elif joint.type_name() in ['revolute', 'continuous']:  ## arm
#             joint.set_default_angle(0)
            joint.set_default_angle(q0[index])
            index += 1
            
        plant.AddJointActuator(joint_name, joint)

   
    return panda_model

def equal(current_q, target_q, epsilon = 0.05):
    if isinstance(target_q, tuple):
        target_q = np.asarray(target_q)

    return np.linalg.norm(current_q - target_q) < epsilon


def dual_drake_ik(xyz_right, quat_right, xyz_left, quat_left, q0_old_ordering, ARM_ONLY=False):
    """ Convert end-effector pose list to joint position list using series of
    InverseKinematics problems. Note that q is 9-dimensional because the last 2 dimensions
    contain gripper joints, but these should not matter to the constraints.
    @param: pose_lst (python list): post_lst[i] contains keyframe X_WG at index i.
    @return: q_knots (python_list): q_knots[i] contains IK solution that will give f(q_knots[i]) \approx pose_lst[i].
    """

    initial_ordering = ['root_x_axis_joint', 
                        'root_y_axis_joint', 
                        'root_z_rotation_joint', 
                        'linear_actuator_height', 
                        'right_panda_joint1', 
                        'right_panda_joint2', 
                        'right_panda_joint3', 
                        'right_panda_joint4', 
                        'right_panda_joint5', 
                        'right_panda_joint6', 
                        'right_panda_joint7', 
                        'right_panda_finger_joint1', 
                        'right_panda_finger_joint2', 
                        'left_panda_joint1', 
                        'left_panda_joint2', 
                        'left_panda_joint3', 
                        'left_panda_joint4', 
                        'left_panda_joint5', 
                        'left_panda_joint6', 
                        'left_panda_joint7', 
                        'left_panda_finger_joint1', 
                        'left_panda_finger_joint2']

    drake_ordering = [  'root_x_axis_joint', 
                        'root_y_axis_joint', 
                        'root_z_rotation_joint', 
                        'linear_actuator_height', 
                        'right_panda_joint1', 'left_panda_joint1',
                        'right_panda_joint2', 'left_panda_joint2',
                        'right_panda_joint3', 'left_panda_joint3',
                        'right_panda_joint4', 'left_panda_joint4', 
                        'right_panda_joint5', 'left_panda_joint5',
                        'right_panda_joint6', 'left_panda_joint6', 
                        'right_panda_joint7', 'left_panda_joint7', 
                        'right_panda_finger_joint1', 
                        'left_panda_finger_joint1',
                        'right_panda_finger_joint2',  
                        'left_panda_finger_joint2']

    # Need to transform from our ordering to drake ordering
    q0 = [0 for _ in range(len(q0_old_ordering))]
    for old_index, joint_value in enumerate(q0_old_ordering):
        joint_name = initial_ordering[old_index]
        q0[drake_ordering.index(joint_name)] = joint_value


    r_right = R.from_quat(quat_right)
    desired_pose_right = RigidTransform(RotationMatrix(r_right.as_matrix()), xyz_right)

    r_left = R.from_quat(quat_left)
    desired_pose_left = RigidTransform(RotationMatrix(r_left.as_matrix()), xyz_left)

    q_knots = []
    plant = CreatePandaControllerPlant(ARM_ONLY=ARM_ONLY, q0=q0, dual=True)
    world_frame = plant.world_frame()
    right_gripper_frame = plant.GetFrameByName("right_panda_link8")
    left_gripper_frame = plant.GetFrameByName("left_panda_link8")

    q_nominal = q0  ##np.array([ -0.93661428, 1.28568379, 0.84716645, -1.98834233, 2.62867444, 1.95018328, -1.50371341]) # nominal joint for joint-centering.
    def AddOrientationConstraint(ik, R_WG_right, R_WG_left, bounds):
        """Add orientation constraint to the ik problem. Implements an inequality
        constraint where the axis-angle difference between f_R(q) and R_WG must be
        within bounds. Can be translated to:
        ik.prog().AddBoundingBoxConstraint(angle_diff(f_R(q), R_WG), -bounds, bounds)
        """
        ik.AddOrientationConstraint(
                frameAbar=world_frame, R_AbarA=R_WG_right,
                frameBbar=right_gripper_frame, R_BbarB=RotationMatrix(),
                theta_bound=bounds
        )
        ik.AddOrientationConstraint(
                frameAbar=world_frame, R_AbarA=R_WG_left,
                frameBbar=left_gripper_frame, R_BbarB=RotationMatrix(),
                theta_bound=bounds
        )

    def AddPositionConstraint(ik, p_WG_lower_right, p_WG_upper_right, p_WG_lower_left, p_WG_upper_left):
        """Add position constraint to the ik problem. Implements an inequality
        constraint where f_p(q) must lie between p_WG_lower and p_WG_upper. Can be
        translated to
        ik.prog().AddBoundingBoxConstraint(f_p(q), p_WG_lower, p_WG_upper)
        """
        ik.AddPositionConstraint(
                frameA=world_frame, frameB=right_gripper_frame, p_BQ=np.zeros(3),
                p_AQ_lower=p_WG_lower_right, p_AQ_upper=p_WG_upper_right)

        ik.AddPositionConstraint(
                frameA=world_frame, frameB=left_gripper_frame, p_BQ=np.zeros(3),
                p_AQ_lower=p_WG_lower_left, p_AQ_upper=p_WG_upper_left)

    ik = inverse_kinematics.InverseKinematics(plant, with_joint_limits=True)
    q_variables = ik.q() # Get variables for MathematicalProgram
    # print(q_variables)
    prog = ik.prog() # Get MathematicalProgram

    #### Modify here ###############################

    X_WG_right = desired_pose_right
    R_WG_right = X_WG_right.rotation()
    p_WG_right = X_WG_right.translation()

    X_WG_left = desired_pose_left
    R_WG_left = X_WG_left.rotation()
    p_WG_left = X_WG_left.translation()


    ## an equality constraint for the constrained degrees of freedom
    AddPositionConstraint(ik, p_WG_right, p_WG_right, p_WG_left, p_WG_left)
    
    ## inequality constraints for the unconstrained one
    AddOrientationConstraint(ik, R_WG_right, R_WG_left, 0.01)

        
    ## Add a joint-centering cost on q_nominal
    prog.AddQuadraticErrorCost(np.identity(len(q_variables)), q_nominal, q_variables)

    ik.AddPointToPointDistanceConstraint(plant.GetFrameByName("right_panda_hand"), [0,0,0], plant.GetFrameByName("adjustable_body"), [0,0,0], 0.5, 1000)
    ik.AddPointToPointDistanceConstraint(plant.GetFrameByName("left_panda_hand"), [0,0,0], plant.GetFrameByName("adjustable_body"), [0,0,0],0.5, 1000)

    ## Set initial guess to be nominal configuration
    prog.SetInitialGuess(q_variables, q_nominal)

    ################################################

    result = Solve(prog)

    if not result.is_success():
        return q0
    
    # print(result.GetSolution(q_variables))
    q_drake_ordering = list(result.GetSolution(q_variables))

    # Now transform back to original ordering
    q = [0 for _ in range(len(q_drake_ordering))]
    for drake_index, joint_value in enumerate(q_drake_ordering):
        joint_name = drake_ordering[drake_index]
        q[initial_ordering.index(joint_name)] = joint_value

    return q

def CreateMultibodyPlantWithPandaController(dimensions, center, q0=None):
    import sys
    if '/opt/drake/lib/python3.8/site-packages' not in sys.path:
        sys.path.append('/opt/drake/lib/python3.8/site-packages')
    from pydrake.all import (
        Adder, AddMultibodyPlantSceneGraph, Demultiplexer, DiagramBuilder,
        InverseDynamicsController, MakeMultibodyStateToWsgStateSystem,
        MeshcatVisualizerCpp, MultibodyPlant, Parser, PassThrough, FindResourceOrThrow,
        SchunkWsgPositionController, StateInterpolatorWithDiscreteDerivative,
        Simulator, Box,
        DiagramBuilder, MeshcatVisualizerCpp, MeshcatVisualizerParams,
        Simulator, FindResourceOrThrow,
        Parser, MultibodyPlant, RigidTransform, RollPitchYaw,
        PiecewisePolynomial, PiecewiseQuaternionSlerp, RotationMatrix, Solve,
        TrajectorySource, ConstantVectorSource
    )
    manipulation = importlib.import_module("drake_examples.manipulation")
    from manipulation.scenarios import AddShape

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    panda = AddPanda(plant, scene_graph, ARM_ONLY=False, q0=q0)
    plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])

    w, l, h = dimensions
    x, y, z = center
    box = AddShape(plant, Box(w, l, h), "box")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("box", box), RigidTransform([x, y, z]))
    plant.Finalize()

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    q0 = plant.GetPositions(plant_context)
    print('len(q0)', len(q0))
    return plant, plant_context

def drake_ik(xyz, quat, q0=None, ARM_ONLY=False, verbose=True,
             hand_link="right_panda_link8", dimensions=[], center=[]):
    """ Convert end-effector pose list to joint position list using series of
    InverseKinematics problems. Note that q is 9-dimensional because the last 2 dimensions
    contain gripper joints, but these should not matter to the constraints.
    @param: pose_lst (python list): post_lst[i] contains keyframe X_WG at index i.
    @return: q_knots (python_list): q_knots[i] contains IK solution that will give f(q_knots[i]) \approx pose_lst[i].
    """

    if isinstance(quat, RotationMatrix):
        r = quat
    else:
        r = R.from_quat(quat)
        r = RotationMatrix(r.as_matrix())

    desired_pose = RigidTransform(r, xyz)

    q_knots = []
    if q0 == None:
        q0 = [ -0.93661428, 1.28568379, 0.84716645, -1.98834233, 2.62867444, 1.95018328, -1.50371341] # nominal joint for joint-centering.
        q0 = [0, 0, 0, 0] + q0 + [0, 0]
        print(len(q0))

    plant = CreatePandaControllerPlant(ARM_ONLY=ARM_ONLY, q0=q0, verbose=False)

    world_frame = plant.world_frame()
    gripper_frame = plant.GetFrameByName(hand_link)
    q_nominal = q0
    # if verbose: print('len(q_nominal)', len(q_nominal))
    def AddOrientationConstraint(ik, R_WG, bounds):
        """Add orientation constraint to the ik problem. Implements an inequality
        constraint where the axis-angle difference between f_R(q) and R_WG must be
        within bounds. Can be translated to:
        ik.prog().AddBoundingBoxConstraint(angle_diff(f_R(q), R_WG), -bounds, bounds)
        """
        ik.AddOrientationConstraint(
                frameAbar=world_frame, R_AbarA=R_WG,
                frameBbar=gripper_frame, R_BbarB=RotationMatrix(),
                theta_bound=bounds
        )

    def AddPositionConstraint(ik, p_WG_lower, p_WG_upper):
        """Add position constraint to the ik problem. Implements an inequality
        constraint where f_p(q) must lie between p_WG_lower and p_WG_upper. Can be
        translated to
        ik.prog().AddBoundingBoxConstraint(f_p(q), p_WG_lower, p_WG_upper)
        """
        ik.AddPositionConstraint(
                frameA=world_frame, frameB=gripper_frame, p_BQ=np.zeros(3),
                p_AQ_lower=p_WG_lower, p_AQ_upper=p_WG_upper)

    ik = inverse_kinematics.InverseKinematics(plant)

    q_variables = ik.q() # Get variables for MathematicalProgram
    prog = ik.prog() # Get MathematicalProgram

    #### Modify here ###############################

    X_WG = desired_pose
    R_WG = X_WG.rotation()
    p_WG = X_WG.translation()

    ## an equality constraint for the constrained degrees of freedom
    AddPositionConstraint(ik, p_WG, p_WG)
    
    ## inequality constraints for the unconstrained one
    AddOrientationConstraint(ik, R_WG, 0.01)
    
    ## Add a joint-centering cost on q_nominal
    prog.AddQuadraticErrorCost(np.identity(len(q_variables)), q_nominal, q_variables)

    ## Set initial guess to be nominal configuration
    prog.SetInitialGuess(q_variables, q_nominal)

    ################################################

    result = Solve(prog)

    if not result.is_success():
        return q0
        
    q = list(result.GetSolution(q_variables))

    return q

if(__name__=="__main__"):
    xyz = [0.5599999998462588, 0.12000000138395996, 0.3799999947283723]
    quat = [ -0.4608969, 0.0483233, 0.7118954, 0.5276779 ]
    print(drake_ik(xyz, quat, ARM_ONLY=False))

