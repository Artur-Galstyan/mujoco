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

import warp as wp

@wp.func
def solve_search_sums(grad: float, solution: float) -> wp.vec2: ...
def create_blocked_cholesky_factorize_solve_func(
  block_size: int, matrix_size_static: int
) -> wp.Function: ...
def create_blocked_cholesky_augmented_factorize_solve_func(
  block_size: int, matrix_size_static: int
) -> wp.Function: ...
def create_blocked_cholesky_augmented_factorize_solve_newton_func(
  block_size: int, matrix_size_static: int, vector_size_static: int
) -> wp.Function: ...
def create_blocked_cholesky_solve_func(
  block_size: int, matrix_size_static: int
) -> wp.Function: ...
def create_blocked_cholesky_solve_newton_func(
  block_size: int, matrix_size_static: int, vector_size_static: int
) -> wp.Function: ...
def ctz64(x: wp.uint64) -> int: ...
def create_blocked_cholesky_sparse_tiles_factorize_solve_newton_func(
  block_size: int, matrix_size_static: int, vector_size_static: int
) -> wp.Function: ...
def create_blocked_cholesky_sparse_tiles_solve_newton_func(
  block_size: int, matrix_size_static: int, vector_size_static: int
) -> wp.Function: ...
