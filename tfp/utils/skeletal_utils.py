from scipy.signal import savgol_filter
import numpy as np

import skeleton_data as sk_data
import prior_sk_data as psk_data
import tfp.config.config as config


def smoothen(preds, window=11, order=3, axis=0):
    return savgol_filter(preds, window, order, axis=axis)

class Transformation:

    def __init__(self,number_joints):
        """
        """
        parent_limbs = 'PARENT_LIMBS_'+str(number_joints)
        self.parent_limbs = config.parent_limbs



    def _gen_limb_graph(self):
        """
        Params:
        li
        """
        n = len(limb_parents)
        G = [[] for i in range(n)]
        for i in range(n):
            j = limb_parents[i]
            if i != j:
                G[j].append(i)
        return G


    def _bfs_order(G, root):
        from collections import deque
        q = deque([root])
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in G[u]:
                q.append(v)
        return order

    def _get_parent_relative_joint_locations(joints_xyz, sk_data=psk_data):
        """
        Assumed shape: [#limbs, X, ...]
        """
        limb_parents = sk_data.limb_parents
        joints_xyz = joints_xyz.swapaxes(0, axis)

        limb_rel_preds = joints_xyz - joints_xyz[limb_parents]
        return limb_rel_preds

    def _get_abs_joint_locations(rel_joints_xyz, axis=0, sk_data=psk_data):
        """
        Assumed shape: [#limbs, X, ...]
        """
        limb_parents, limb_order = sk_data.limb_parents, sk_data.limb_order

        abs_preds = np.zeros_like(rel_joints_xyz, dtype=np.float64)
        abs_preds[limb_order[0]] = rel_joints_xyz[limb_order[0]]
        for l in limb_order[1:]:
            p = limb_parents[l]
            abs_preds[l] = abs_preds[p] + rel_joints_xyz[l]
        return abs_preds


    def _cart2sph(xyz):
        """
        Assumed shape: [#limbs,3]
        """
        # Hypotenuse of the triangle formed in xy plane.It is required
        # to calculate the azimuth angle(φ)
        hxy = np.hypot(xyz[:,0],xyz[:,1])
        # Spherical coordinates matrix
        rtp = np.zeros_like(xyz, dtype=np.float64)
        # Radial distance(r)
        rtp[:,0] = np.linalg.norm(xyz, axis=1)
        # Polar angle(θ)
        rtp[:,1] = np.arctan2(hxy,xyz[:,2])
        # Azimuth angle(φ)
        rtp[:,2] = np.arctan2(xyz[:,1],xyz[:,0])
        return rtp


    def _sph2cart(rtp):
        """
        Assumed shape: [#limbs,3]
        """
        # Cartesian coordinates matrix
        xyz = np.zeros_like(rtp, dtype=np.float64)
        # x coordinates: r * sin(θ) * cos(φ)
        xyz[:,0] = rtp[:,0] * np.sin(rtp[:,1]) * np.cos(rtp[:,2])
        # y coordinates: r * sin(θ) * sin(φ)
        xyz[1] = rtp[0] * np.sin(rtp[1]) * np.sin(rtp[2])
        # z coordinates: r * cos(θ)
        xyz[2] = rtp[0] * np.sin(rtp[1])
        return xyz


    def transform(preds, head_length=2.0, sk_data=psk_data):
        """
        :param preds: joint values of a skeleton or skeleton vector
        :shape: [#limbs, 3, X, X, ...]
        """
        rel_preds = get_parent_relative_joint_locations(preds, sk_data=sk_data)

        sph_rel_preds = cart2sph(rel_preds)

        fixed_limb_lengths = head_length * sk_data.limb_ratios

        sph_rel_preds[:, 0] = fixed_limb_lengths

        cart_rel_preds = sph2cart(sph_rel_preds)

        cart_abs_preds = get_abs_joint_locations(cart_rel_preds, sk_data=sk_data)

        return cart_abs_preds


    def fit_skeleton_frames(preds, head_length=2.0, sk_data=psk_data):
        return fit_skeleton_frame(preds.transpose([1, 2, 0]) , head_length, sk_data).transpose([2, 0, 1])


    def scale_local_skeleton(preds, head_length=2.0):
        fixed_limb_lengths = np.expand_dims(head_length * psk_data.limb_ratios, axis=1)
        return preds * fixed_limb_lengths
