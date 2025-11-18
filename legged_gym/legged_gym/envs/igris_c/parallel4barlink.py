import math
import torch


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternions to 3x3 rotation matrices.

    q: (..., 4) [x, y, z, w]
    returns: (..., 3, 3)
    """
    x, y, z, w = q.unbind(-1)

    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w

    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    # Row-major 3x3
    m00 = ww + xx - yy - zz
    m01 = 2 * (xy - zw)
    m02 = 2 * (xz + yw)

    m10 = 2 * (xy + zw)
    m11 = ww - xx + yy - zz
    m12 = 2 * (yz - xw)

    m20 = 2 * (xz - yw)
    m21 = 2 * (yz + xw)
    m22 = ww - xx - yy + zz

    return torch.stack([
        torch.stack([m00, m01, m02], dim=-1),
        torch.stack([m10, m11, m12], dim=-1),
        torch.stack([m20, m21, m22], dim=-1),
    ], dim=-2)


class Parallel4BarLinkTorch:
    """
    Torch version of your Parallel4BarLink, with batch support.

    Conventions:
    - All tensors are torch.float32.
    - Shapes:
        r_a_init, r_b_init, r_c_init: (3, 2)
        joint / motor angles: (B, 2)           where B = num_envs
        r_c (current C positions): (B, 3, 2)
        jac_joint: (B, 6, 2)   (6D twist Jacobian at waist; linear stacked above angular)
    """

    def __init__(
        self,
        r_a_init: torch.Tensor,
        r_b_init: torch.Tensor,
        r_c_init: torch.Tensor,
        motor_angles_min: torch.Tensor,
        motor_angles_max: torch.Tensor,
        joint_angles_min: torch.Tensor,
        joint_angles_max: torch.Tensor,
        is_elbow_up: bool = True,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Parameters
        ----------
        r_a_init, r_b_init, r_c_init : (3, 2) tensors (base frame)
        motor_angles_min/max : (2,) tensors
        joint_angles_min/max : (2,) tensors
        is_elbow_up : bool (same for both sides; extend to per-side if needed)
        """

        self.device = torch.device(device)
        self.dtype = dtype

        self.r_a_init = r_a_init.to(self.device, self.dtype)      # (3,2)
        self.r_b_init = r_b_init.to(self.device, self.dtype)      # (3,2)
        self.r_c_init = r_c_init.to(self.device, self.dtype)      # (3,2)
        self.r_c_offset_local = self.r_c_init.clone()
        # constant geometry
        self.b_vec = (self.r_b_init - self.r_a_init)              # (3,2)
        self.l_bar = torch.norm(self.b_vec, dim=0)                # (2,)
        self.l_rod = torch.norm(self.r_c_init - self.r_b_init, dim=0)  # (2,)

        # limits
        self.motor_angles_min = motor_angles_min.to(self.device, self.dtype)  # (2,)
        self.motor_angles_max = motor_angles_max.to(self.device, self.dtype)  # (2,)
        self.joint_angles_min = joint_angles_min.to(self.device, self.dtype)  # (2,)
        self.joint_angles_max = joint_angles_max.to(self.device, self.dtype)  # (2,)

        self.is_elbow_up = is_elbow_up

    # ------------------------------------------------------------------
    # Inverse kinematics: joint angles -> motor angles
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ik(self, joint_angles: torch.Tensor, r_c: torch.Tensor) -> torch.Tensor:
        """
        Four-bar IK, batched.

        joint_angles : (B, 2)  [roll, pitch] commands (radians)
        r_c          : (B, 3, 2) current C positions in base frame for both linkages

        Returns
        -------
        motor_angles : (B, 2)
        """
        device = self.device
        dtype = self.dtype

        q = joint_angles.to(device, dtype)
        B = q.shape[0]

        # Clamp joint angles to limits (like C++ code)
        q = torch.max(torch.min(q, self.joint_angles_max.view(1, 2)), 
                      self.joint_angles_min.view(1, 2))

        # Output
        motor_angles = torch.zeros_like(q)

        # A and B initial geometry
        r_a = self.r_a_init  # (3,2)
        b_vec = self.b_vec   # (3,2)

        for i in range(2):
            # a_vec = r_c - r_a_init
            a_vec = r_c[:, :, i] - r_a[:, i].view(1, 3)   # (B,3)
            b = b_vec[:, i]                               # (3,)
            l_bar_i = self.l_bar[i]
            l_rod_i = self.l_rod[i]

            ax = a_vec[:, 0]
            ay = a_vec[:, 1]
            az = a_vec[:, 2]

            bx = b[0]
            by = b[1]
            bz = b[2]

            # d, e as in C++ code
            a_sq = ax * ax + ay * ay + az * az
            d = -(l_rod_i * l_rod_i - l_bar_i * l_bar_i - a_sq) / 2.0
            e = d - ay * by

            # A, B, C coefficients (be careful with names vs matrix A,B,C)
            A_coef = (ax * ax + az * az) * (bx * bx + bz * bz)
            B_coef = (ax * bz - az * bx) * e
            C_coef = (
                e * e - (
                    ax * ax * bx * bx
                    + az * az * bz * bz
                    + 2.0 * ax * az * bx * bz
                )
            )

            disc = B_coef * B_coef - A_coef * C_coef
            disc = torch.clamp(disc, min=0.0)
            sqrt_disc = torch.sqrt(disc)

            # value_pos/neg = sin(theta) candidates
            # clip to [-1,1] like IgrisMath::minmax_cut
            val_pos = (B_coef + sqrt_disc) / A_coef
            val_neg = (B_coef - sqrt_disc) / A_coef
            val_pos = torch.clamp(val_pos, -1.0, 1.0)
            val_neg = torch.clamp(val_neg, -1.0, 1.0)

            motor_angle_pos = torch.asin(val_pos)  # (B,)
            motor_angle_neg = torch.asin(val_neg)  # (B,)

            # six candidates, like C++:
            # pos, neg,  π-pos,  π-neg,  -π-pos,  -π-neg
            cand = torch.stack([
                motor_angle_pos,
                motor_angle_neg,
                math.pi - motor_angle_pos,
                math.pi - motor_angle_neg,
                -math.pi - motor_angle_pos,
                -math.pi - motor_angle_neg,
            ], dim=1)   # (B,6)

            # Within motor limits
            mmin = self.motor_angles_min[i]
            mmax = self.motor_angles_max[i]
            within_limits = (cand >= mmin) & (cand <= mmax)   # (B,6)

            # Now we apply the geometric checks for each candidate:
            # - |r_c - r_b| ≈ l_rod
            # - elbow direction (bar × rod).y has correct sign
            r_a_i = r_a[:, i]  # (3,)

            valid = torch.zeros_like(within_limits)

            for k in range(6):
                theta_k = cand[:, k]   # (B,)

                cos_t = torch.cos(theta_k)
                sin_t = torch.sin(theta_k)

                # Rotate initial bar vector around global y by theta:
                # r_bar = R_y(theta) * b_vec
                r_bar_x = bx * cos_t + bz * sin_t
                r_bar_y = by * torch.ones_like(theta_k)
                r_bar_z = -bx * sin_t + bz * cos_t
                r_bar = torch.stack([r_bar_x, r_bar_y, r_bar_z], dim=1)  # (B,3)

                r_b = r_a_i.view(1, 3) + r_bar                            # (B,3)
                r_c_i = r_c[:, :, i]                                      # (B,3)
                r_rod = r_c_i - r_b                                       # (B,3)

                rod_len = torch.norm(r_rod, dim=1)
                rod_len_ok = torch.abs(rod_len - l_rod_i) < 1e-4          # (B,)

                elbow_vec = torch.cross(r_bar, r_rod, dim=1)              # (B,3)
                elbow_up = elbow_vec[:, 1] > 0.0                          # y-component

                if self.is_elbow_up:
                    elbow_ok = elbow_up
                else:
                    elbow_ok = ~elbow_up

                valid_k = within_limits[:, k] & rod_len_ok & elbow_ok
                valid[:, k] = valid_k

            # Pick the first valid candidate per env (if any)
            has_valid = valid.any(dim=1)        # (B,)
            # argmax on bool will give index of first True (or 0 if none)
            idx = torch.argmax(valid.int(), dim=1)  # (B,)

            chosen = cand[torch.arange(B, device=device), idx]  # (B,)

            # Fallback for rare "no valid candidate" cases:
            # clamp motor_angle_pos just to stay sane
            fallback = torch.clamp(motor_angle_pos, mmin, mmax)
            chosen = torch.where(has_valid, chosen, fallback)

            motor_angles[:, i] = chosen

        # Extra safety clamp
        motor_angles = torch.max(
            torch.min(motor_angles, self.motor_angles_max.view(1, 2)),
            self.motor_angles_min.view(1, 2),
        )

        return motor_angles

    # ------------------------------------------------------------------
    # Jacobian: J_c = dtheta/dq (joint velocity → motor velocity)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def jac(
        self,
        joint_angles: torch.Tensor,
        motor_angles: torch.Tensor,
        r_c: torch.Tensor,
        jac_joint: torch.Tensor,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        """
        Compute J_c = dtheta/dq, using your analytic formulation:

            J_x   from rod and r_c
            J_theta from (r_bar x r_rod).y
            J     = J_theta^{-1} * J_x
            J_c   = J * jac_joint

        Parameters
        ----------
        joint_angles : (B, 2)   (unused except for consistency; kept for interface)
        motor_angles : (B, 2)   current motor angles (output of ik)
        r_c          : (B, 3, 2) current C positions in base frame
        jac_joint    : (B, 6, 2) 6D Jacobian at waist link:
                        [linear; angular] w.r.t. the same 2 waist DOFs.

        Returns
        -------
        J_c : (B, 2, 2)
              J_c[i] maps qdot -> thetadot for env i.
        """
        device = self.device
        dtype = self.dtype

        q = joint_angles.to(device, dtype)
        theta = motor_angles.to(device, dtype)
        r_c = r_c.to(device, dtype)
        jac_joint = jac_joint.to(device, dtype)

        B = q.shape[0]

        # Geometry
        r_a = self.r_a_init  # (3,2)
        b_vec = self.b_vec   # (3,2)

        # Outputs
        J_x = torch.zeros(B, 2, 6, device=device, dtype=dtype)
        J_theta_diag = torch.zeros(B, 2, device=device, dtype=dtype)

        for i in range(2):
            theta_i = theta[:, i]                 # (B,)
            bx, by, bz = b_vec[:, i]             # scalars
            r_a_i = r_a[:, i]                    # (3,)

            cos_t = torch.cos(theta_i)
            sin_t = torch.sin(theta_i)

            # r_bar = R_y(theta) * b_vec (same as in IK)
            r_bar_x = bx * cos_t + bz * sin_t
            r_bar_y = by * torch.ones_like(theta_i)
            r_bar_z = -bx * sin_t + bz * cos_t
            r_bar = torch.stack([r_bar_x, r_bar_y, r_bar_z], dim=1)  # (B,3)

            r_b = r_a_i.view(1, 3) + r_bar                           # (B,3)
            r_c_i = r_c[:, :, i]                                     # (B,3)
            r_rod = r_c_i - r_b                                      # (B,3)

            # J_x rows
            # J_x_.block(0,0,1,3) = r_rod^T
            # J_x_.block(0,3,1,3) = (r_c x r_rod)^T
            cross_c_rod = torch.cross(r_c_i, r_rod, dim=1)           # (B,3)
            J_x[:, i, 0:3] = r_rod
            J_x[:, i, 3:6] = cross_c_rod

            # J_theta_(i,i) = (r_bar x r_rod)(1)
            cross_bar_rod = torch.cross(r_bar, r_rod, dim=1)         # (B,3)
            J_theta_diag[:, i] = cross_bar_rod[:, 1]                 # y-component

        # Avoid division by zero in J_theta inverse
        d = J_theta_diag.clone()          # (B,2)
        d_abs = torch.abs(d)
        sign = torch.sign(d)
        d_safe = torch.where(d_abs < eps, sign * eps, d)  # (B,2)

        # J = J_theta^{-1} * J_x : row-wise division by diag entries
        J = J_x / d_safe.unsqueeze(-1)    # (B,2,1) -> (B,2,6)

        # Finally: J_c = J * jac_joint  (2x6)*(6x2) = 2x2, batched
        # jac_joint is expected as [linear; angular], like after the row-swap in C++
        J_c = torch.bmm(J, jac_joint)     # (B,2,2)

        return J_c
