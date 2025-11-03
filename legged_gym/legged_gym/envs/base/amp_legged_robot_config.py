# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class AMPLeggedRobotCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        reference_state_initialization = True
        reference_state_initialization_prob = .8
        amp_motion_files = {
            # "path/to/pkl": {
            #     "hz": motion_hz,
            #     "start_time": clip_motion_start [s],
            #     "end_time": clip_motion_end [s],
            #     "weight": motion_weight
            # },
        }
        amp_preload_transitions = True
        amp_num_preload_transitions = 2000000

class AMPLeggedRobotCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = 'HIMAMPOnPolicyRunner'
        
    class algorithm(LeggedRobotCfgPPO.algorithm):
        amp_replay_buffer_size = 100000
        disc_coef = 1.
        disc_grad_pen = 1.

    class runner(LeggedRobotCfgPPO.runner):
        algorithm_class_name = 'HIMAMPPPO'
        amp_reward_coef = 3.0 * (AMPLeggedRobotCfg.sim.dt * AMPLeggedRobotCfg.control.decimation)
        amp_motion_files = AMPLeggedRobotCfg.env.amp_motion_files
        amp_num_preload_transitions = AMPLeggedRobotCfg.env.amp_num_preload_transitions
        amp_task_reward_lerp = .3
        amp_discr_hidden_dims = [128,]