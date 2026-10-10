# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import ctypes
from collections.abc import Iterable, Sequence
from typing import IO, TypeAlias

import numpy as np
from numpy import typing as npt

from mujoco import _specs, _structs
from mujoco._callbacks import get_mjcb_act_bias as get_mjcb_act_bias
from mujoco._callbacks import get_mjcb_act_dyn as get_mjcb_act_dyn
from mujoco._callbacks import get_mjcb_act_gain as get_mjcb_act_gain
from mujoco._callbacks import get_mjcb_contactfilter as get_mjcb_contactfilter
from mujoco._callbacks import get_mjcb_control as get_mjcb_control
from mujoco._callbacks import get_mjcb_passive as get_mjcb_passive
from mujoco._callbacks import get_mjcb_sensor as get_mjcb_sensor
from mujoco._callbacks import get_mjcb_time as get_mjcb_time
from mujoco._callbacks import get_mju_user_free as get_mju_user_free
from mujoco._callbacks import get_mju_user_malloc as get_mju_user_malloc
from mujoco._callbacks import get_mju_user_warning as get_mju_user_warning
from mujoco._callbacks import set_mjcb_act_bias as set_mjcb_act_bias
from mujoco._callbacks import set_mjcb_act_dyn as set_mjcb_act_dyn
from mujoco._callbacks import set_mjcb_act_gain as set_mjcb_act_gain
from mujoco._callbacks import set_mjcb_contactfilter as set_mjcb_contactfilter
from mujoco._callbacks import set_mjcb_control as set_mjcb_control
from mujoco._callbacks import set_mjcb_passive as set_mjcb_passive
from mujoco._callbacks import set_mjcb_sensor as set_mjcb_sensor
from mujoco._callbacks import set_mjcb_time as set_mjcb_time
from mujoco._callbacks import set_mju_user_free as set_mju_user_free
from mujoco._callbacks import set_mju_user_malloc as set_mju_user_malloc
from mujoco._callbacks import set_mju_user_warning as set_mju_user_warning
from mujoco._constants import MJTNUM_BYTES as MJTNUM_BYTES
from mujoco._constants import mjDISABLESTRING as mjDISABLESTRING
from mujoco._constants import mjENABLESTRING as mjENABLESTRING
from mujoco._constants import mjFRAMESTRING as mjFRAMESTRING
from mujoco._constants import mjLABELSTRING as mjLABELSTRING
from mujoco._constants import mjMAXCONPAIR as mjMAXCONPAIR
from mujoco._constants import mjMAXFLEXNODES as mjMAXFLEXNODES
from mujoco._constants import mjMAXIMP as mjMAXIMP
from mujoco._constants import mjMAXLIGHT as mjMAXLIGHT
from mujoco._constants import mjMAXLINE as mjMAXLINE
from mujoco._constants import mjMAXLINEPNT as mjMAXLINEPNT
from mujoco._constants import mjMAXOVERLAY as mjMAXOVERLAY
from mujoco._constants import mjMAXPLANEGRID as mjMAXPLANEGRID
from mujoco._constants import mjMAXTREEDEPTH as mjMAXTREEDEPTH
from mujoco._constants import mjMAXVAL as mjMAXVAL
from mujoco._constants import mjMINAWAKE as mjMINAWAKE
from mujoco._constants import mjMINIMP as mjMINIMP
from mujoco._constants import mjMINMU as mjMINMU
from mujoco._constants import mjMINVAL as mjMINVAL
from mujoco._constants import mjNBIAS as mjNBIAS
from mujoco._constants import mjNDYN as mjNDYN
from mujoco._constants import mjNEQDATA as mjNEQDATA
from mujoco._constants import mjNFLUID as mjNFLUID
from mujoco._constants import mjNGAIN as mjNGAIN
from mujoco._constants import mjNGROUP as mjNGROUP
from mujoco._constants import mjNIMP as mjNIMP
from mujoco._constants import mjNISLAND as mjNISLAND
from mujoco._constants import mjNPOLY as mjNPOLY
from mujoco._constants import mjNREF as mjNREF
from mujoco._constants import mjNSENS as mjNSENS
from mujoco._constants import mjNSOLVER as mjNSOLVER
from mujoco._constants import mjPI as mjPI
from mujoco._constants import mjRNDSTRING as mjRNDSTRING
from mujoco._constants import mjTIMERSTRING as mjTIMERSTRING
from mujoco._constants import mjVERSION_HEADER as mjVERSION_HEADER
from mujoco._constants import mjVISSTRING as mjVISSTRING
from mujoco._enums import mjrIndexType as mjrIndexType
from mujoco._enums import mjrMeshPrimitiveType as mjrMeshPrimitiveType
from mujoco._enums import mjrPixelFormat as mjrPixelFormat
from mujoco._enums import mjrVertexAttributeType as mjrVertexAttributeType
from mujoco._enums import mjrVertexAttributeUsage as mjrVertexAttributeUsage
from mujoco._enums import mjtActuator as mjtActuator
from mujoco._enums import mjtAlignFree as mjtAlignFree
from mujoco._enums import mjtBias as mjtBias
from mujoco._enums import mjtBuiltin as mjtBuiltin
from mujoco._enums import mjtButton as mjtButton
from mujoco._enums import mjtCamera as mjtCamera
from mujoco._enums import mjtCamLight as mjtCamLight
from mujoco._enums import mjtCamOutBit as mjtCamOutBit
from mujoco._enums import mjtCatBit as mjtCatBit
from mujoco._enums import mjtColorSpace as mjtColorSpace
from mujoco._enums import mjtConDataField as mjtConDataField
from mujoco._enums import mjtCone as mjtCone
from mujoco._enums import mjtConflict as mjtConflict
from mujoco._enums import mjtConstraint as mjtConstraint
from mujoco._enums import mjtConstraintState as mjtConstraintState
from mujoco._enums import mjtCTimer as mjtCTimer
from mujoco._enums import mjtCtrlChart as mjtCtrlChart
from mujoco._enums import mjtCtrlInput as mjtCtrlInput
from mujoco._enums import mjtDataType as mjtDataType
from mujoco._enums import mjtDepthMap as mjtDepthMap
from mujoco._enums import mjtDisableBit as mjtDisableBit
from mujoco._enums import mjtDyn as mjtDyn
from mujoco._enums import mjtEnableBit as mjtEnableBit
from mujoco._enums import mjtEq as mjtEq
from mujoco._enums import mjtEvent as mjtEvent
from mujoco._enums import mjtFlexSelf as mjtFlexSelf
from mujoco._enums import mjtFont as mjtFont
from mujoco._enums import mjtFontScale as mjtFontScale
from mujoco._enums import mjtFrame as mjtFrame
from mujoco._enums import mjtFramebuffer as mjtFramebuffer
from mujoco._enums import mjtGain as mjtGain
from mujoco._enums import mjtGeom as mjtGeom
from mujoco._enums import mjtGeomInertia as mjtGeomInertia
from mujoco._enums import mjtGridPos as mjtGridPos
from mujoco._enums import mjtInertiaFromGeom as mjtInertiaFromGeom
from mujoco._enums import mjtIntegrator as mjtIntegrator
from mujoco._enums import mjtItem as mjtItem
from mujoco._enums import mjtJacobian as mjtJacobian
from mujoco._enums import mjtJoint as mjtJoint
from mujoco._enums import mjtLabel as mjtLabel
from mujoco._enums import mjtLightType as mjtLightType
from mujoco._enums import mjtLimited as mjtLimited
from mujoco._enums import mjtLogLevel as mjtLogLevel
from mujoco._enums import mjtLogTopic as mjtLogTopic
from mujoco._enums import mjtLRMode as mjtLRMode
from mujoco._enums import mjtMark as mjtMark
from mujoco._enums import mjtMeshBuiltin as mjtMeshBuiltin
from mujoco._enums import mjtMeshInertia as mjtMeshInertia
from mujoco._enums import mjtMouse as mjtMouse
from mujoco._enums import mjtObj as mjtObj
from mujoco._enums import mjtOrientation as mjtOrientation
from mujoco._enums import mjtPertBit as mjtPertBit
from mujoco._enums import mjtPluginCapabilityBit as mjtPluginCapabilityBit
from mujoco._enums import mjtProjection as mjtProjection
from mujoco._enums import mjtRayDataField as mjtRayDataField
from mujoco._enums import mjtRndFlag as mjtRndFlag
from mujoco._enums import mjtSameFrame as mjtSameFrame
from mujoco._enums import mjtSDFType as mjtSDFType
from mujoco._enums import mjtSection as mjtSection
from mujoco._enums import mjtSensor as mjtSensor
from mujoco._enums import mjtSleepPolicy as mjtSleepPolicy
from mujoco._enums import mjtSleepState as mjtSleepState
from mujoco._enums import mjtSolver as mjtSolver
from mujoco._enums import mjtStage as mjtStage
from mujoco._enums import mjtState as mjtState
from mujoco._enums import mjtStereo as mjtStereo
from mujoco._enums import mjtTexture as mjtTexture
from mujoco._enums import mjtTextureRole as mjtTextureRole
from mujoco._enums import mjtTimer as mjtTimer
from mujoco._enums import mjtTrn as mjtTrn
from mujoco._enums import mjtVisFlag as mjtVisFlag
from mujoco._enums import mjtWarning as mjtWarning
from mujoco._enums import mjtWrap as mjtWrap
from mujoco._errors import FatalError as FatalError
from mujoco._errors import UnexpectedError as UnexpectedError
from mujoco._functions import mj_actuatorInputName as mj_actuatorInputName
from mujoco._functions import mj_addContact as mj_addContact
from mujoco._functions import mj_addM as mj_addM
from mujoco._functions import mj_angmomMat as mj_angmomMat
from mujoco._functions import mj_applyFT as mj_applyFT
from mujoco._functions import mj_camlight as mj_camlight
from mujoco._functions import mj_checkAcc as mj_checkAcc
from mujoco._functions import mj_checkPos as mj_checkPos
from mujoco._functions import mj_checkVel as mj_checkVel
from mujoco._functions import mj_clearCache as mj_clearCache
from mujoco._functions import mj_collision as mj_collision
from mujoco._functions import mj_compareFwdInv as mj_compareFwdInv
from mujoco._functions import mj_comPos as mj_comPos
from mujoco._functions import mj_comVel as mj_comVel
from mujoco._functions import mj_constraintUpdate as mj_constraintUpdate
from mujoco._functions import mj_contactForce as mj_contactForce
from mujoco._functions import mj_copyData as mj_copyData
from mujoco._functions import mj_copyState as mj_copyState
from mujoco._functions import mj_crb as mj_crb
from mujoco._functions import mj_defaultLROpt as mj_defaultLROpt
from mujoco._functions import mj_defaultOption as mj_defaultOption
from mujoco._functions import mj_defaultSolRefImp as mj_defaultSolRefImp
from mujoco._functions import mj_defaultVisual as mj_defaultVisual
from mujoco._functions import mj_differentiatePos as mj_differentiatePos
from mujoco._functions import mj_energyPos as mj_energyPos
from mujoco._functions import mj_energyVel as mj_energyVel
from mujoco._functions import mj_Euler as mj_Euler
from mujoco._functions import mj_extractState as mj_extractState
from mujoco._functions import mj_factorM as mj_factorM
from mujoco._functions import mj_flex as mj_flex
from mujoco._functions import mj_forward as mj_forward
from mujoco._functions import mj_forwardSkip as mj_forwardSkip
from mujoco._functions import mj_fullM as mj_fullM
from mujoco._functions import mj_fwdAcceleration as mj_fwdAcceleration
from mujoco._functions import mj_fwdActuation as mj_fwdActuation
from mujoco._functions import mj_fwdConstraint as mj_fwdConstraint
from mujoco._functions import mj_fwdKinematics as mj_fwdKinematics
from mujoco._functions import mj_fwdPosition as mj_fwdPosition
from mujoco._functions import mj_fwdVelocity as mj_fwdVelocity
from mujoco._functions import mj_geomDistance as mj_geomDistance
from mujoco._functions import mj_getCache as mj_getCache
from mujoco._functions import mj_getCacheCapacity as mj_getCacheCapacity
from mujoco._functions import mj_getCacheSize as mj_getCacheSize
from mujoco._functions import mj_getState as mj_getState
from mujoco._functions import mj_getTotalmass as mj_getTotalmass
from mujoco._functions import mj_id2name as mj_id2name
from mujoco._functions import mj_implicit as mj_implicit
from mujoco._functions import mj_initCtrlHistory as mj_initCtrlHistory
from mujoco._functions import mj_initSensorHistory as mj_initSensorHistory
from mujoco._functions import mj_insideSite as mj_insideSite
from mujoco._functions import mj_integratePos as mj_integratePos
from mujoco._functions import mj_invConstraint as mj_invConstraint
from mujoco._functions import mj_inverse as mj_inverse
from mujoco._functions import mj_inverseSkip as mj_inverseSkip
from mujoco._functions import mj_invPosition as mj_invPosition
from mujoco._functions import mj_invVelocity as mj_invVelocity
from mujoco._functions import mj_isDual as mj_isDual
from mujoco._functions import mj_island as mj_island
from mujoco._functions import mj_isPyramidal as mj_isPyramidal
from mujoco._functions import mj_isSparse as mj_isSparse
from mujoco._functions import mj_jac as mj_jac
from mujoco._functions import mj_jacBody as mj_jacBody
from mujoco._functions import mj_jacBodyCom as mj_jacBodyCom
from mujoco._functions import mj_jacDot as mj_jacDot
from mujoco._functions import mj_jacGeom as mj_jacGeom
from mujoco._functions import mj_jacPointAxis as mj_jacPointAxis
from mujoco._functions import mj_jacSite as mj_jacSite
from mujoco._functions import mj_jacSubtreeCom as mj_jacSubtreeCom
from mujoco._functions import mj_kinematics as mj_kinematics
from mujoco._functions import (
  mj_loadAllPluginLibraries as mj_loadAllPluginLibraries,
)
from mujoco._functions import mj_loadPluginLibrary as mj_loadPluginLibrary
from mujoco._functions import mj_local2Global as mj_local2Global
from mujoco._functions import mj_makeConstraint as mj_makeConstraint
from mujoco._functions import mj_makeM as mj_makeM
from mujoco._functions import mj_maxContact as mj_maxContact
from mujoco._functions import mj_mulJacTVec as mj_mulJacTVec
from mujoco._functions import mj_mulJacVec as mj_mulJacVec
from mujoco._functions import mj_mulM as mj_mulM
from mujoco._functions import mj_mulM2 as mj_mulM2
from mujoco._functions import mj_multiRay as mj_multiRay
from mujoco._functions import mj_name2id as mj_name2id
from mujoco._functions import mj_normalizeQuat as mj_normalizeQuat
from mujoco._functions import mj_objectAcceleration as mj_objectAcceleration
from mujoco._functions import mj_objectVelocity as mj_objectVelocity
from mujoco._functions import mj_passive as mj_passive
from mujoco._functions import mj_printData as mj_printData
from mujoco._functions import mj_printFormattedData as mj_printFormattedData
from mujoco._functions import mj_printFormattedModel as mj_printFormattedModel
from mujoco._functions import mj_printFormattedScene as mj_printFormattedScene
from mujoco._functions import mj_printModel as mj_printModel
from mujoco._functions import mj_printScene as mj_printScene
from mujoco._functions import mj_printSchema as mj_printSchema
from mujoco._functions import mj_projectConstraint as mj_projectConstraint
from mujoco._functions import mj_ray as mj_ray
from mujoco._functions import mj_rayFlex as mj_rayFlex
from mujoco._functions import mj_rayHfield as mj_rayHfield
from mujoco._functions import mj_rayMesh as mj_rayMesh
from mujoco._functions import mj_readCtrl as mj_readCtrl
from mujoco._functions import mj_readSensor as mj_readSensor
from mujoco._functions import mj_referenceConstraint as mj_referenceConstraint
from mujoco._functions import mj_resetCallbacks as mj_resetCallbacks
from mujoco._functions import mj_resetCtrl as mj_resetCtrl
from mujoco._functions import mj_resetData as mj_resetData
from mujoco._functions import mj_resetDataDebug as mj_resetDataDebug
from mujoco._functions import mj_resetDataKeyframe as mj_resetDataKeyframe
from mujoco._functions import mj_rne as mj_rne
from mujoco._functions import mj_rnePostConstraint as mj_rnePostConstraint
from mujoco._functions import mj_RungeKutta as mj_RungeKutta
from mujoco._functions import mj_saveLastXML as mj_saveLastXML
from mujoco._functions import mj_saveModel as mj_saveModel
from mujoco._functions import mj_sensorAcc as mj_sensorAcc
from mujoco._functions import mj_sensorPos as mj_sensorPos
from mujoco._functions import mj_sensorVel as mj_sensorVel
from mujoco._functions import mj_setCacheCapacity as mj_setCacheCapacity
from mujoco._functions import mj_setConst as mj_setConst
from mujoco._functions import mj_setKeyframe as mj_setKeyframe
from mujoco._functions import mj_setLengthRange as mj_setLengthRange
from mujoco._functions import mj_setState as mj_setState
from mujoco._functions import mj_setTotalmass as mj_setTotalmass
from mujoco._functions import mj_sizeModel as mj_sizeModel
from mujoco._functions import mj_solveM as mj_solveM
from mujoco._functions import mj_solveM2 as mj_solveM2
from mujoco._functions import mj_stateSize as mj_stateSize
from mujoco._functions import mj_step as mj_step
from mujoco._functions import mj_step1 as mj_step1
from mujoco._functions import mj_step2 as mj_step2
from mujoco._functions import mj_subtreeVel as mj_subtreeVel
from mujoco._functions import mj_tendon as mj_tendon
from mujoco._functions import mj_transmission as mj_transmission
from mujoco._functions import mj_version as mj_version
from mujoco._functions import mj_versionString as mj_versionString
from mujoco._functions import mjd_inverseFD as mjd_inverseFD
from mujoco._functions import mjd_quatIntegrate as mjd_quatIntegrate
from mujoco._functions import mjd_subQuat as mjd_subQuat
from mujoco._functions import mjd_transitionFD as mjd_transitionFD
from mujoco._functions import mju_add as mju_add
from mujoco._functions import mju_add3 as mju_add3
from mujoco._functions import mju_addScl as mju_addScl
from mujoco._functions import mju_addScl3 as mju_addScl3
from mujoco._functions import mju_addTo as mju_addTo
from mujoco._functions import mju_addTo3 as mju_addTo3
from mujoco._functions import mju_addToScl as mju_addToScl
from mujoco._functions import mju_addToScl3 as mju_addToScl3
from mujoco._functions import mju_axisAngle2Quat as mju_axisAngle2Quat
from mujoco._functions import mju_band2Dense as mju_band2Dense
from mujoco._functions import mju_bandDiag as mju_bandDiag
from mujoco._functions import mju_bandMulMatVec as mju_bandMulMatVec
from mujoco._functions import mju_boxQP as mju_boxQP
from mujoco._functions import mju_cholFactor as mju_cholFactor
from mujoco._functions import mju_cholFactorBand as mju_cholFactorBand
from mujoco._functions import mju_cholSolve as mju_cholSolve
from mujoco._functions import mju_cholSolveBand as mju_cholSolveBand
from mujoco._functions import mju_cholUpdate as mju_cholUpdate
from mujoco._functions import mju_clip as mju_clip
from mujoco._functions import mju_copy as mju_copy
from mujoco._functions import mju_copy3 as mju_copy3
from mujoco._functions import mju_copy4 as mju_copy4
from mujoco._functions import mju_cross as mju_cross
from mujoco._functions import mju_d2n as mju_d2n
from mujoco._functions import mju_decodePyramid as mju_decodePyramid
from mujoco._functions import mju_dense2Band as mju_dense2Band
from mujoco._functions import mju_dense2sparse as mju_dense2sparse
from mujoco._functions import mju_derivQuat as mju_derivQuat
from mujoco._functions import mju_dist3 as mju_dist3
from mujoco._functions import mju_dot as mju_dot
from mujoco._functions import mju_dot3 as mju_dot3
from mujoco._functions import mju_eig3 as mju_eig3
from mujoco._functions import mju_encodePyramid as mju_encodePyramid
from mujoco._functions import mju_euler2Quat as mju_euler2Quat
from mujoco._functions import mju_eye as mju_eye
from mujoco._functions import mju_f2n as mju_f2n
from mujoco._functions import mju_fill as mju_fill
from mujoco._functions import mju_getXMLDependencies as mju_getXMLDependencies
from mujoco._functions import mju_Halton as mju_Halton
from mujoco._functions import mju_insertionSort as mju_insertionSort
from mujoco._functions import mju_insertionSortInt as mju_insertionSortInt
from mujoco._functions import mju_isBad as mju_isBad
from mujoco._functions import mju_isZero as mju_isZero
from mujoco._functions import mju_L1 as mju_L1
from mujoco._functions import mju_mat2Quat as mju_mat2Quat
from mujoco._functions import mju_mat2Rot as mju_mat2Rot
from mujoco._functions import mju_max as mju_max
from mujoco._functions import mju_min as mju_min
from mujoco._functions import mju_mulMatMat as mju_mulMatMat
from mujoco._functions import mju_mulMatMatT as mju_mulMatMatT
from mujoco._functions import mju_mulMatTMat as mju_mulMatTMat
from mujoco._functions import mju_mulMatTVec as mju_mulMatTVec
from mujoco._functions import mju_mulMatTVec3 as mju_mulMatTVec3
from mujoco._functions import mju_mulMatVec as mju_mulMatVec
from mujoco._functions import mju_mulMatVec3 as mju_mulMatVec3
from mujoco._functions import mju_mulPose as mju_mulPose
from mujoco._functions import mju_mulQuat as mju_mulQuat
from mujoco._functions import mju_mulQuatAxis as mju_mulQuatAxis
from mujoco._functions import mju_mulVecMatVec as mju_mulVecMatVec
from mujoco._functions import mju_muscleBias as mju_muscleBias
from mujoco._functions import mju_muscleDynamics as mju_muscleDynamics
from mujoco._functions import mju_muscleGain as mju_muscleGain
from mujoco._functions import mju_n2d as mju_n2d
from mujoco._functions import mju_n2f as mju_n2f
from mujoco._functions import mju_negPose as mju_negPose
from mujoco._functions import mju_negQuat as mju_negQuat
from mujoco._functions import mju_norm as mju_norm
from mujoco._functions import mju_norm3 as mju_norm3
from mujoco._functions import mju_normalize as mju_normalize
from mujoco._functions import mju_normalize3 as mju_normalize3
from mujoco._functions import mju_normalize4 as mju_normalize4
from mujoco._functions import mju_printMat as mju_printMat
from mujoco._functions import mju_printMatSparse as mju_printMatSparse
from mujoco._functions import mju_quat2Mat as mju_quat2Mat
from mujoco._functions import mju_quat2Vel as mju_quat2Vel
from mujoco._functions import mju_quatIntegrate as mju_quatIntegrate
from mujoco._functions import mju_quatZ2Vec as mju_quatZ2Vec
from mujoco._functions import mju_rayGeom as mju_rayGeom
from mujoco._functions import mju_raySkin as mju_raySkin
from mujoco._functions import mju_rotVecQuat as mju_rotVecQuat
from mujoco._functions import mju_round as mju_round
from mujoco._functions import mju_scl as mju_scl
from mujoco._functions import mju_scl3 as mju_scl3
from mujoco._functions import mju_sigmoid as mju_sigmoid
from mujoco._functions import mju_sign as mju_sign
from mujoco._functions import mju_sparse2dense as mju_sparse2dense
from mujoco._functions import mju_springDamper as mju_springDamper
from mujoco._functions import mju_sqrMatTD as mju_sqrMatTD
from mujoco._functions import mju_standardNormal as mju_standardNormal
from mujoco._functions import mju_str2Type as mju_str2Type
from mujoco._functions import mju_sub as mju_sub
from mujoco._functions import mju_sub3 as mju_sub3
from mujoco._functions import mju_subFrom as mju_subFrom
from mujoco._functions import mju_subFrom3 as mju_subFrom3
from mujoco._functions import mju_subQuat as mju_subQuat
from mujoco._functions import mju_sum as mju_sum
from mujoco._functions import mju_sym2dense as mju_sym2dense
from mujoco._functions import mju_symmetrize as mju_symmetrize
from mujoco._functions import mju_threadpool as mju_threadpool
from mujoco._functions import mju_transformSpatial as mju_transformSpatial
from mujoco._functions import mju_transpose as mju_transpose
from mujoco._functions import mju_trnVecPose as mju_trnVecPose
from mujoco._functions import mju_type2Str as mju_type2Str
from mujoco._functions import mju_unit4 as mju_unit4
from mujoco._functions import mju_warningText as mju_warningText
from mujoco._functions import mju_writeLog as mju_writeLog
from mujoco._functions import mju_writeNumBytes as mju_writeNumBytes
from mujoco._functions import mju_zero as mju_zero
from mujoco._functions import mju_zero3 as mju_zero3
from mujoco._functions import mju_zero4 as mju_zero4
from mujoco._functions import mjv_addGeoms as mjv_addGeoms
from mujoco._functions import mjv_alignToCamera as mjv_alignToCamera
from mujoco._functions import mjv_applyPerturbForce as mjv_applyPerturbForce
from mujoco._functions import mjv_applyPerturbPose as mjv_applyPerturbPose
from mujoco._functions import mjv_cameraFrame as mjv_cameraFrame
from mujoco._functions import mjv_cameraFrustum as mjv_cameraFrustum
from mujoco._functions import mjv_cameraInModel as mjv_cameraInModel
from mujoco._functions import mjv_cameraInRoom as mjv_cameraInRoom
from mujoco._functions import mjv_connector as mjv_connector
from mujoco._functions import mjv_defaultCamera as mjv_defaultCamera
from mujoco._functions import mjv_defaultFigure as mjv_defaultFigure
from mujoco._functions import mjv_defaultFreeCamera as mjv_defaultFreeCamera
from mujoco._functions import mjv_defaultOption as mjv_defaultOption
from mujoco._functions import mjv_defaultPerturb as mjv_defaultPerturb
from mujoco._functions import mjv_frustumHeight as mjv_frustumHeight
from mujoco._functions import mjv_initGeom as mjv_initGeom
from mujoco._functions import mjv_initPerturb as mjv_initPerturb
from mujoco._functions import mjv_makeLights as mjv_makeLights
from mujoco._functions import mjv_model2room as mjv_model2room
from mujoco._functions import mjv_moveCamera as mjv_moveCamera
from mujoco._functions import mjv_moveModel as mjv_moveModel
from mujoco._functions import mjv_movePerturb as mjv_movePerturb
from mujoco._functions import mjv_room2model as mjv_room2model
from mujoco._functions import mjv_select as mjv_select
from mujoco._functions import mjv_updateCamera as mjv_updateCamera
from mujoco._functions import mjv_updateScene as mjv_updateScene
from mujoco._functions import mjv_updateSkin as mjv_updateSkin
from mujoco._render import MjrContext as MjrContext
from mujoco._render import MjrRendererInfo as MjrRendererInfo
from mujoco._render import mjr_addAux as mjr_addAux
from mujoco._render import mjr_blitAux as mjr_blitAux
from mujoco._render import mjr_blitBuffer as mjr_blitBuffer
from mujoco._render import mjr_changeFont as mjr_changeFont
from mujoco._render import mjr_defaultRendererInfo as mjr_defaultRendererInfo
from mujoco._render import mjr_drawPixels as mjr_drawPixels
from mujoco._render import mjr_figure as mjr_figure
from mujoco._render import mjr_findRect as mjr_findRect
from mujoco._render import mjr_finish as mjr_finish
from mujoco._render import mjr_getError as mjr_getError
from mujoco._render import mjr_getRendererInfo as mjr_getRendererInfo
from mujoco._render import mjr_label as mjr_label
from mujoco._render import mjr_maxViewport as mjr_maxViewport
from mujoco._render import mjr_overlay as mjr_overlay
from mujoco._render import mjr_readPixels as mjr_readPixels
from mujoco._render import mjr_rectangle as mjr_rectangle
from mujoco._render import mjr_render as mjr_render
from mujoco._render import mjr_resizeOffscreen as mjr_resizeOffscreen
from mujoco._render import mjr_restoreBuffer as mjr_restoreBuffer
from mujoco._render import mjr_setAux as mjr_setAux
from mujoco._render import mjr_setBuffer as mjr_setBuffer
from mujoco._render import mjr_text as mjr_text
from mujoco._render import mjr_uploadHField as mjr_uploadHField
from mujoco._render import mjr_uploadMesh as mjr_uploadMesh
from mujoco._render import mjr_uploadTexture as mjr_uploadTexture
from mujoco._specs import MjByteVec as MjByteVec
from mujoco._specs import MjCharVec as MjCharVec
from mujoco._specs import MjDoubleVec as MjDoubleVec
from mujoco._specs import MjFloatVec as MjFloatVec
from mujoco._specs import MjIntVec as MjIntVec
from mujoco._specs import MjsActuator as MjsActuator
from mujoco._specs import MjsAuthored as MjsAuthored
from mujoco._specs import MjsBody as MjsBody
from mujoco._specs import MjsCamera as MjsCamera
from mujoco._specs import MjsCompiler as MjsCompiler
from mujoco._specs import MjsDefault as MjsDefault
from mujoco._specs import MjsElement as MjsElement
from mujoco._specs import MjsEquality as MjsEquality
from mujoco._specs import MjsExclude as MjsExclude
from mujoco._specs import MjsFlex as MjsFlex
from mujoco._specs import MjsFrame as MjsFrame
from mujoco._specs import MjsGeom as MjsGeom
from mujoco._specs import MjsHField as MjsHField
from mujoco._specs import MjsJoint as MjsJoint
from mujoco._specs import MjsKey as MjsKey
from mujoco._specs import MjsLight as MjsLight
from mujoco._specs import MjsMaterial as MjsMaterial
from mujoco._specs import MjsMesh as MjsMesh
from mujoco._specs import MjsNumeric as MjsNumeric
from mujoco._specs import MjsOrientation as MjsOrientation
from mujoco._specs import MjsPair as MjsPair
from mujoco._specs import MjSpec as MjSpec
from mujoco._specs import MjsPlugin as MjsPlugin
from mujoco._specs import MjsSensor as MjsSensor
from mujoco._specs import MjsSite as MjsSite
from mujoco._specs import MjsSkin as MjsSkin
from mujoco._specs import MjsTendon as MjsTendon
from mujoco._specs import MjsTendonPath as MjsTendonPath
from mujoco._specs import MjsText as MjsText
from mujoco._specs import MjsTexture as MjsTexture
from mujoco._specs import MjStringVec as MjStringVec
from mujoco._specs import MjsTuple as MjsTuple
from mujoco._specs import MjsWrap as MjsWrap
from mujoco._specs import MjVfs as MjVfs
from mujoco._specs import MjVisualHeadlight as MjVisualHeadlight
from mujoco._specs import MjVisualRgba as MjVisualRgba
from mujoco._structs import MjContact as MjContact
from mujoco._structs import MjData as MjData
from mujoco._structs import MjLogConfig as MjLogConfig
from mujoco._structs import MjLogMessage as MjLogMessage
from mujoco._structs import MjLROpt as MjLROpt
from mujoco._structs import MjModel as MjModel
from mujoco._structs import MjOption as MjOption
from mujoco._structs import MjPreContact as MjPreContact
from mujoco._structs import MjrRect as MjrRect
from mujoco._structs import MjrVertexAttribute as MjrVertexAttribute
from mujoco._structs import MjSolverStat as MjSolverStat
from mujoco._structs import MjStatistic as MjStatistic
from mujoco._structs import MjTimerStat as MjTimerStat
from mujoco._structs import MjvCamera as MjvCamera
from mujoco._structs import MjvFigure as MjvFigure
from mujoco._structs import MjvGeom as MjvGeom
from mujoco._structs import MjvGLCamera as MjvGLCamera
from mujoco._structs import MjVisual as MjVisual
from mujoco._structs import MjvLight as MjvLight
from mujoco._structs import MjvOption as MjvOption
from mujoco._structs import MjvPerturb as MjvPerturb
from mujoco._structs import MjvScene as MjvScene
from mujoco._structs import MjWarningStat as MjWarningStat
from mujoco._structs import mjv_averageCamera as mjv_averageCamera
from mujoco._structs import mjv_camera2GLCamera as mjv_camera2GLCamera
from mujoco.rendering.classic.gl_context import GLContext as GLContext
from mujoco.rendering.classic.renderer import Renderer as Renderer

MjStruct: TypeAlias = (
  _specs.MjsBody
  | _specs.MjsFrame
  | _specs.MjsGeom
  | _specs.MjsJoint
  | _specs.MjsLight
  | _specs.MjsMaterial
  | _specs.MjsSite
  | _specs.MjsMesh
  | _specs.MjsSkin
  | _specs.MjsTexture
  | _specs.MjsText
  | _specs.MjsTuple
  | _specs.MjsCamera
  | _specs.MjsFlex
  | _specs.MjsHField
  | _specs.MjsKey
  | _specs.MjsNumeric
  | _specs.MjsPair
  | _specs.MjsExclude
  | _specs.MjsEquality
  | _specs.MjsTendon
  | _specs.MjsSensor
  | _specs.MjsActuator
  | _specs.MjsPlugin
)

MJTNUM_DTYPE: type[np.float32 | np.float64]
__version__: str
HEADERS_DIR: str
PLUGINS_DIR: str
PLUGIN_HANDLES: list[ctypes.CDLL]

def from_zip(file: str | IO[bytes]) -> MjSpec: ...
def to_zip(spec: MjSpec, file: str | IO[bytes]) -> None: ...

class _MjBindModel:
  elements: Sequence[
    _structs._MjModelActuatorViews
    | _structs._MjModelBodyViews
    | _structs._MjModelCameraViews
    | _structs._MjModelEqualityViews
    | _structs._MjModelExcludeViews
    | _structs._MjModelGeomViews
    | _structs._MjModelHfieldViews
    | _structs._MjModelJointViews
    | _structs._MjModelLightViews
    | _structs._MjModelMaterialViews
    | _structs._MjModelMeshViews
    | _structs._MjModelNumericViews
    | _structs._MjModelPairViews
    | _structs._MjModelSensorViews
    | _structs._MjModelSiteViews
    | _structs._MjModelSkinViews
    | _structs._MjModelTendonViews
    | _structs._MjModelTextureViews
    | _structs._MjModelTupleViews
    | _structs._MjModelKeyframeViews
  ]
  @property
  def J_colind(self) -> list[np.int32]: ...
  @property
  def J_rowadr(self) -> list[np.int32]: ...
  @property
  def J_rownnz(self) -> list[np.int32]: ...
  @property
  def M0(self) -> list[np.float32 | np.float64]: ...
  @property
  def Madr(self) -> list[np.int32]: ...
  @property
  def _adr(self) -> list[np.int32]: ...
  @property
  def _damping(self) -> list[np.float32 | np.float64]: ...
  @property
  def _dampingpoly(self) -> list[np.float32 | np.float64]: ...
  @property
  def _frictionloss(self) -> list[np.float32 | np.float64]: ...
  @property
  def _group(self) -> list[np.int32]: ...
  @property
  def _invweight0(self) -> list[np.float32 | np.float64]: ...
  @property
  def _length0(self) -> list[np.float32 | np.float64]: ...
  @property
  def _lengthspring(self) -> list[np.float32 | np.float64]: ...
  @property
  def _limited(self) -> list[np.uint8]: ...
  @property
  def _margin(self) -> list[np.float32 | np.float64]: ...
  @property
  def _matid(self) -> list[np.int32]: ...
  @property
  def _num(self) -> list[np.int32]: ...
  @property
  def _range(self) -> list[np.float32 | np.float64]: ...
  @property
  def _rgba(self) -> list[np.float32]: ...
  @property
  def _solimp_fri(self) -> list[np.float32 | np.float64]: ...
  @property
  def _solimp_lim(self) -> list[np.float32 | np.float64]: ...
  @property
  def _solref_fri(self) -> list[np.float32 | np.float64]: ...
  @property
  def _solref_lim(self) -> list[np.float32 | np.float64]: ...
  @property
  def _stiffness(self) -> list[np.float32 | np.float64]: ...
  @property
  def _stiffnesspoly(self) -> list[np.float32 | np.float64]: ...
  @property
  def _user(self) -> list[np.float32 | np.float64]: ...
  @property
  def _width(self) -> list[np.float32 | np.float64]: ...
  @property
  def acc0(self) -> list[np.float32 | np.float64]: ...
  @property
  def act(self) -> list[np.float32 | np.float64]: ...
  @property
  def actadr(self) -> list[np.int32]: ...
  @property
  def active(self) -> list[np.uint8]: ...
  @property
  def active0(self) -> list[np.uint8]: ...
  @property
  def actlimited(self) -> list[np.uint8]: ...
  @property
  def actnum(self) -> list[np.int32]: ...
  @property
  def actrange(self) -> list[np.float32 | np.float64]: ...
  @property
  def adr(self) -> list[np.int32 | np.int64]: ...
  @property
  def ambient(self) -> list[np.float32]: ...
  @property
  def armature(self) -> list[np.float32 | np.float64]: ...
  @property
  def attenuation(self) -> list[np.float32]: ...
  @property
  def axis(self) -> list[np.float32 | np.float64]: ...
  @property
  def biasprm(self) -> list[np.float32 | np.float64]: ...
  @property
  def biastype(self) -> list[np.int32]: ...
  @property
  def bodyid(self) -> list[np.int32]: ...
  @property
  def boneadr(self) -> list[np.int32]: ...
  @property
  def bonenum(self) -> list[np.int32]: ...
  @property
  def castshadow(self) -> list[np.uint8]: ...
  @property
  def conaffinity(self) -> list[np.int32]: ...
  @property
  def condim(self) -> list[np.int32]: ...
  @property
  def contype(self) -> list[np.int32]: ...
  @property
  def cranklength(self) -> list[np.float32 | np.float64]: ...
  @property
  def ctrl(self) -> list[np.float32 | np.float64]: ...
  @property
  def ctrllimited(self) -> list[np.uint8]: ...
  @property
  def ctrlrange(self) -> list[np.float32 | np.float64]: ...
  @property
  def cutoff(self) -> list[np.float32 | np.float64]: ...
  @property
  def damping(self) -> list[np.float32 | np.float64]: ...
  @property
  def dampingpoly(
    self,
  ) -> list[npt.NDArray[np.float32] | npt.NDArray[np.float64]]: ...
  @property
  def data(
    self,
  ) -> list[
    np.float32 | np.float64 | npt.NDArray[np.float32] | npt.NDArray[np.uint8]
  ]: ...
  @property
  def dataid(self) -> list[np.int32]: ...
  @property
  def datatype(self) -> list[np.int32]: ...
  @property
  def diffuse(self) -> list[np.float32]: ...
  @property
  def dim(self) -> list[np.int32]: ...
  @property
  def dir(self) -> list[np.float32 | np.float64]: ...
  @property
  def dir0(self) -> list[np.float32 | np.float64]: ...
  @property
  def dofadr(self) -> list[np.int32]: ...
  @property
  def dofnum(self) -> list[np.int32]: ...
  @property
  def dynprm(self) -> list[np.float32 | np.float64]: ...
  @property
  def dyntype(self) -> list[np.int32]: ...
  @property
  def emission(self) -> list[np.float32]: ...
  @property
  def exponent(self) -> list[np.float32]: ...
  @property
  def faceadr(self) -> list[np.int32]: ...
  @property
  def facenum(self) -> list[np.int32]: ...
  @property
  def forcelimited(self) -> list[np.uint8]: ...
  @property
  def forcerange(self) -> list[np.float32 | np.float64]: ...
  @property
  def fovy(self) -> list[np.float32 | np.float64]: ...
  @property
  def friction(self) -> list[np.float32 | np.float64]: ...
  @property
  def frictionloss(self) -> list[np.float32 | np.float64]: ...
  @property
  def gainprm(self) -> list[np.float32 | np.float64]: ...
  @property
  def gaintype(self) -> list[np.int32]: ...
  @property
  def gap(self) -> list[np.float32 | np.float64]: ...
  @property
  def gear(self) -> list[np.float32 | np.float64]: ...
  @property
  def geom1(self) -> list[np.int32]: ...
  @property
  def geom2(self) -> list[np.int32]: ...
  @property
  def geomadr(self) -> list[np.int32]: ...
  @property
  def geomnum(self) -> list[np.int32]: ...
  @property
  def graphadr(self) -> list[np.int32]: ...
  @property
  def group(self) -> list[np.int32]: ...
  @property
  def height(self) -> list[np.int32]: ...
  @property
  def inertia(self) -> list[np.float32 | np.float64]: ...
  @property
  def inflate(self) -> list[np.float32]: ...
  @property
  def invweight0(self) -> list[np.float32 | np.float64]: ...
  @property
  def ipd(self) -> list[np.float32 | np.float64]: ...
  @property
  def ipos(self) -> list[np.float32 | np.float64]: ...
  @property
  def iquat(self) -> list[np.float32 | np.float64]: ...
  @property
  def jntadr(self) -> list[np.int32]: ...
  @property
  def jntid(self) -> list[np.int32]: ...
  @property
  def jntnum(self) -> list[np.int32]: ...
  @property
  def length0(self) -> list[np.float32 | np.float64]: ...
  @property
  def lengthrange(self) -> list[np.float32 | np.float64]: ...
  @property
  def limited(self) -> list[np.uint8]: ...
  @property
  def margin(self) -> list[np.float32 | np.float64]: ...
  @property
  def mass(self) -> list[np.float32 | np.float64]: ...
  @property
  def mat0(self) -> list[np.float32 | np.float64]: ...
  @property
  def matid(self) -> list[np.int32]: ...
  @property
  def mocapid(self) -> list[np.int32]: ...
  @property
  def mode(self) -> list[np.int32]: ...
  @property
  def mpos(self) -> list[np.float32 | np.float64]: ...
  @property
  def mquat(self) -> list[np.float32 | np.float64]: ...
  @property
  def nchannel(self) -> list[np.int32]: ...
  @property
  def ncol(self) -> list[np.int32]: ...
  @property
  def needstage(self) -> list[np.int32]: ...
  @property
  def noise(self) -> list[np.float32 | np.float64]: ...
  @property
  def nrow(self) -> list[np.int32]: ...
  @property
  def obj1id(self) -> list[np.int32]: ...
  @property
  def obj2id(self) -> list[np.int32]: ...
  @property
  def objid(self) -> list[np.int32]: ...
  @property
  def objprm(self) -> list[np.float32 | np.float64]: ...
  @property
  def objtype(self) -> list[np.int32]: ...
  @property
  def parentid(self) -> list[np.int32]: ...
  @property
  def pos(self) -> list[np.float32 | np.float64]: ...
  @property
  def pos0(self) -> list[np.float32 | np.float64]: ...
  @property
  def poscom0(self) -> list[np.float32 | np.float64]: ...
  @property
  def priority(self) -> list[np.int32]: ...
  @property
  def qpos(self) -> list[np.float32 | np.float64]: ...
  @property
  def qpos0(self) -> list[np.float32 | np.float64]: ...
  @property
  def qpos_spring(self) -> list[np.float32 | np.float64]: ...
  @property
  def qposadr(self) -> list[np.int32]: ...
  @property
  def quat(self) -> list[np.float32 | np.float64]: ...
  @property
  def qvel(self) -> list[np.float32 | np.float64]: ...
  @property
  def range(self) -> list[np.float32 | np.float64]: ...
  @property
  def rbound(self) -> list[np.float32 | np.float64]: ...
  @property
  def refid(self) -> list[np.int32]: ...
  @property
  def reflectance(self) -> list[np.float32]: ...
  @property
  def reftype(self) -> list[np.int32]: ...
  @property
  def rgba(self) -> list[np.float32]: ...
  @property
  def rootid(self) -> list[np.int32]: ...
  @property
  def sameframe(self) -> list[np.uint8]: ...
  @property
  def shininess(self) -> list[np.float32]: ...
  @property
  def signature(self) -> list[np.int32]: ...
  @property
  def simple(self) -> list[np.uint8]: ...
  @property
  def simplenum(self) -> list[np.int32]: ...
  @property
  def size(self) -> list[np.float32 | np.float64 | np.int32]: ...
  @property
  def solimp(self) -> list[np.float32 | np.float64]: ...
  @property
  def solmix(self) -> list[np.float32 | np.float64]: ...
  @property
  def solref(self) -> list[np.float32 | np.float64]: ...
  @property
  def specular(self) -> list[np.float32]: ...
  @property
  def stiffness(self) -> list[np.float32 | np.float64]: ...
  @property
  def stiffnesspoly(self) -> list[np.float32 | np.float64]: ...
  @property
  def subtreemass(self) -> list[np.float32 | np.float64]: ...
  @property
  def targetbodyid(self) -> list[np.int32]: ...
  @property
  def texcoordadr(self) -> list[np.int32]: ...
  @property
  def texid(self) -> list[np.int32]: ...
  @property
  def texrepeat(self) -> list[np.float32]: ...
  @property
  def texuniform(self) -> list[np.uint8]: ...
  @property
  def time(self) -> list[np.float32 | np.float64]: ...
  @property
  def trnid(self) -> list[np.int32]: ...
  @property
  def trntype(self) -> list[np.int32]: ...
  @property
  def type(self) -> list[np.int32]: ...
  @property
  def user(self) -> list[np.float32 | np.float64]: ...
  @property
  def vertadr(self) -> list[np.int32]: ...
  @property
  def vertnum(self) -> list[np.int32]: ...
  @property
  def weldid(self) -> list[np.int32]: ...
  @property
  def width(self) -> list[np.int32]: ...

class _MjBindData:
  elements: Sequence[
    _structs._MjDataActuatorViews
    | _structs._MjDataBodyViews
    | _structs._MjDataCameraViews
    | _structs._MjDataGeomViews
    | _structs._MjDataJointViews
    | _structs._MjDataLightViews
    | _structs._MjDataSensorViews
    | _structs._MjDataSiteViews
    | _structs._MjDataTendonViews
  ]
  @property
  def cacc(self) -> list[np.float32 | np.float64]: ...
  @cacc.setter
  def cacc(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def cdof(self) -> list[npt.NDArray[np.float32] | npt.NDArray[np.float64]]: ...
  @cdof.setter
  def cdof(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def cdof_dot(
    self,
  ) -> list[npt.NDArray[np.float32] | npt.NDArray[np.float64]]: ...
  @cdof_dot.setter
  def cdof_dot(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def cfrc_ext(self) -> list[np.float32 | np.float64]: ...
  @cfrc_ext.setter
  def cfrc_ext(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def cfrc_int(self) -> list[np.float32 | np.float64]: ...
  @cfrc_int.setter
  def cfrc_int(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def cinert(self) -> list[np.float32 | np.float64]: ...
  @cinert.setter
  def cinert(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def crb(self) -> list[np.float32 | np.float64]: ...
  @crb.setter
  def crb(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def ctrl(self) -> list[np.float32 | np.float64]: ...
  @ctrl.setter
  def ctrl(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def cvel(self) -> list[np.float32 | np.float64]: ...
  @cvel.setter
  def cvel(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def data(self) -> list[np.float32 | np.float64]: ...
  @data.setter
  def data(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def force(self) -> list[np.float32 | np.float64]: ...
  @force.setter
  def force(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def length(self) -> list[np.float32 | np.float64]: ...
  @length.setter
  def length(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def moment(self) -> list[np.float32 | np.float64]: ...
  @moment.setter
  def moment(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def qLDiagInv(self) -> list[np.float32 | np.float64]: ...
  @qLDiagInv.setter
  def qLDiagInv(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def qacc(self) -> list[np.float32 | np.float64]: ...
  @qacc.setter
  def qacc(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def qacc_smooth(self) -> list[np.float32 | np.float64]: ...
  @qacc_smooth.setter
  def qacc_smooth(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def qacc_warmstart(self) -> list[np.float32 | np.float64]: ...
  @qacc_warmstart.setter
  def qacc_warmstart(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def qfrc_actuator(self) -> list[np.float32 | np.float64]: ...
  @qfrc_actuator.setter
  def qfrc_actuator(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def qfrc_applied(self) -> list[np.float32 | np.float64]: ...
  @qfrc_applied.setter
  def qfrc_applied(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def qfrc_bias(self) -> list[np.float32 | np.float64]: ...
  @qfrc_bias.setter
  def qfrc_bias(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def qfrc_constraint(self) -> list[np.float32 | np.float64]: ...
  @qfrc_constraint.setter
  def qfrc_constraint(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def qfrc_inverse(self) -> list[np.float32 | np.float64]: ...
  @qfrc_inverse.setter
  def qfrc_inverse(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def qfrc_passive(self) -> list[np.float32 | np.float64]: ...
  @qfrc_passive.setter
  def qfrc_passive(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def qfrc_smooth(self) -> list[np.float32 | np.float64]: ...
  @qfrc_smooth.setter
  def qfrc_smooth(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def qpos(self) -> list[np.float32 | np.float64]: ...
  @qpos.setter
  def qpos(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def qvel(self) -> list[np.float32 | np.float64]: ...
  @qvel.setter
  def qvel(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def subtree_angmom(self) -> list[np.float32 | np.float64]: ...
  @subtree_angmom.setter
  def subtree_angmom(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def subtree_com(self) -> list[np.float32 | np.float64]: ...
  @subtree_com.setter
  def subtree_com(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def subtree_linvel(self) -> list[np.float32 | np.float64]: ...
  @subtree_linvel.setter
  def subtree_linvel(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def velocity(self) -> list[np.float32 | np.float64]: ...
  @velocity.setter
  def velocity(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def wrapadr(self) -> list[np.int32]: ...
  @wrapadr.setter
  def wrapadr(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def wrapnum(self) -> list[np.int32]: ...
  @wrapnum.setter
  def wrapnum(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def xanchor(self) -> list[np.float32 | np.float64]: ...
  @xanchor.setter
  def xanchor(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def xaxis(self) -> list[np.float32 | np.float64]: ...
  @xaxis.setter
  def xaxis(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def xdir(self) -> list[np.float32 | np.float64]: ...
  @xdir.setter
  def xdir(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def xfrc_applied(self) -> list[np.float32 | np.float64]: ...
  @xfrc_applied.setter
  def xfrc_applied(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def ximat(self) -> list[np.float32 | np.float64]: ...
  @ximat.setter
  def ximat(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def xipos(self) -> list[np.float32 | np.float64]: ...
  @xipos.setter
  def xipos(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def xmat(self) -> list[np.float32 | np.float64]: ...
  @xmat.setter
  def xmat(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def xpos(self) -> list[np.float32 | np.float64]: ...
  @xpos.setter
  def xpos(self, value: Iterable[npt.ArrayLike]) -> None: ...
  @property
  def xquat(self) -> list[np.float32 | np.float64]: ...
  @xquat.setter
  def xquat(self, value: Iterable[npt.ArrayLike]) -> None: ...
