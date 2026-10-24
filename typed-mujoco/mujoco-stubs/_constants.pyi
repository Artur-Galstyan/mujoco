# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

__all__: list[str] = [
  'MJTNUM_BYTES',
  'mjDISABLESTRING',
  'mjENABLESTRING',
  'mjFRAMESTRING',
  'mjLABELSTRING',
  'mjMAXCONPAIR',
  'mjMAXFLEXNODES',
  'mjMAXIMP',
  'mjMAXLIGHT',
  'mjMAXLINE',
  'mjMAXLINEPNT',
  'mjMAXOVERLAY',
  'mjMAXPLANEGRID',
  'mjMAXTREEDEPTH',
  'mjMAXVAL',
  'mjMINAWAKE',
  'mjMINIMP',
  'mjMINMU',
  'mjMINVAL',
  'mjNBIAS',
  'mjNDYN',
  'mjNEQDATA',
  'mjNFLUID',
  'mjNGAIN',
  'mjNGROUP',
  'mjNIMP',
  'mjNISLAND',
  'mjNPOLY',
  'mjNREF',
  'mjNSENS',
  'mjNSOLVER',
  'mjPI',
  'mjRNDSTRING',
  'mjSC_IN',
  'mjSC_INOUT',
  'mjSC_OUT',
  'mjTIMERSTRING',
  'mjVERSION_HEADER',
  'mjVISSTRING',
]
MJTNUM_BYTES: int = 8
mjDISABLESTRING: tuple = ...
mjENABLESTRING: tuple = (
  'Override',
  'Energy',
  'Fwdinv',
  'InvDiscrete',
  'Sleep',
  'DiagExact',
  'IPC',
)
mjFRAMESTRING: tuple = (
  'None',
  'Body',
  'Geom',
  'Site',
  'Camera',
  'Light',
  'Contact',
  'World',
)
mjLABELSTRING: tuple = ...
mjMAXCONPAIR: int = 50
mjMAXFLEXNODES: int = 27
mjMAXIMP: float = 0.9999
mjMAXLIGHT: int = 100
mjMAXLINE: int = 100
mjMAXLINEPNT: int = 1001
mjMAXOVERLAY: int = 500
mjMAXPLANEGRID: int = 200
mjMAXTREEDEPTH: int = 50
mjMAXVAL: float = ...
mjMINAWAKE: int = 10
mjMINIMP: float = 0.0001
mjMINMU: float = 1e-05
mjMINVAL: float = 1e-15
mjNBIAS: int = 10
mjNDYN: int = 10
mjNEQDATA: int = 12
mjNFLUID: int = 12
mjNGAIN: int = 10
mjNGROUP: int = 6
mjNIMP: int = 5
mjNISLAND: int = 20
mjNPOLY: int = 2
mjNREF: int = 2
mjNSENS: int = 3
mjNSOLVER: int = 200
mjPI: float = ...
mjRNDSTRING: tuple = ...
mjSC_IN: int = 1
mjSC_INOUT: int = 3
mjSC_OUT: int = 2
mjTIMERSTRING: tuple = ...
mjVERSION_HEADER: int = 3017000
mjVISSTRING: tuple = ...
