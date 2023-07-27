import sys
from ll4ma_isaacgym.core import Simulator, OfflineSimulatorResponse
from ll4ma_util import ui_util

import torch


class OfflineSimulator:
    """
    Offline simulator for computing things in batch separate from the main sim,
    e.g. forward kinematics. Written for communicating over multiprocessing pipes
    so it can run in its own process concurrently with another sim (since Isaac
    Gym can't run two instances of the simulator in one process).
    """

    def __init__(self, session_config):
        self.config = session_config
        self.sim = Simulator(self.config)
        
    def run(self, pipe):
        """
        Main entry point for running offline sim. Waits indefinitely for a request
        to be written to the pipe, processes the request, and writes the response
        back to the pipe.
        """
        self.dof_state = self.sim.get_dof_state() # (n_envs, n_dofs, 2), last dim pos/vel
        
        while True:
            req = pipe.recv()
            resp = OfflineSimulatorResponse()
            if req.fk_joint_pos is not None:
                resp.fk_poses = self.forward_kinematics(req.fk_joint_pos)
            if req.act_joint_pos is not None:
                self.simulate(req, resp)
            pipe.send(resp)
        
    def forward_kinematics(self, joint_pos):
        """
        Computes forward kinematics in batch.

        Args:
            joint_pos (Tensor): Tensor of joint positions (n_batch, n_joints)
        """
        if joint_pos.size(1) != self.dof_state.size(1):
            ui_util.print_error(f"Joint state in FK request of size {joint_pos.size(1)} "
                                f"but expected size {self.dof_state.size(1)}")
            return None
        
        batches = torch.split(joint_pos, self.config.n_envs)
        poses = []
        for batch in batches:
            n_batch = batch.size(0)
            self._set_batch_dof_state(batch)
            ee_state = self.sim.get_ee_state() # (n_envs, 13), last dim pos(3)/quat(4)/vel(6)
            poses.append(ee_state[:n_batch,:7])
        return torch.cat(poses, dim=0)

    def simulate(self, req, resp):
        # Assume act_joint_pos is shape (n_timesteps, n_samples, n_joints)
        batches = torch.split(req.act_joint_pos, self.config.n_envs, dim=1) # Split along sample dim

        # Create a tensor of the joint position to initialize every environment the same
        joint_pos = req.state.joint_position.view(1, self.dof_state.size(1))
        joint_pos = joint_pos.repeat(self.config.n_envs, 1)

        poses = []
        for batch in batches:
            n_steps, n_batch, n_joints = batch.size()
            self._set_batch_dof_state(joint_pos)
            batch_poses = []
            batch_contacts = [[] for _ in range(n_batch)]
            for i in range(n_steps):
                self.sim.set_target_joint_position_tensor(batch[i].squeeze())
                self.sim.step()
                if req.get_forward_kinematics:
                    ee_state = self.sim.get_ee_state() # (n_envs, 13), last dim pos(3)/quat(4)/vel(6)
                    batch_poses.append(ee_state[:n_batch,:7])
                if req.get_contacts:
                    for env_idx in range(n_batch):
                        env_step_contacts = self.sim.get_env_contacts(
                            env_idx, exclude_pairs=req.contact_exclude_pairs)
                        batch_contacts[env_idx].append(env_step_contacts)
            poses.append(torch.stack(batch_poses, dim=0))
            if req.get_contacts:
                resp.contacts += batch_contacts
            
        if req.get_forward_kinematics:
            resp.fk_poses = torch.cat(poses, dim=1) # (n_steps, n_samples, 7)

    def _set_batch_dof_state(self, batch):
        n_batch = batch.size(0)
        self.dof_state[:n_batch,:,0] = batch
        self.sim.set_dof_state(self.dof_state)
        self.sim.refresh_rigid_body_state()
        



