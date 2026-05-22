import torch
import torch.nn as nn
import numpy as np
from lib.utils.general_utils import quaternion_raw_multiply, get_expon_lr_func, quaternion_slerp, quaternion_raw_multiply_theta
from lib.config import cfg
from lib.utils.camera_utils import Camera

class ActorPose(nn.Module):      
    def __init__(self, tracklets, tracklet_timestamps, camera_timestamps, obj_info):
        # tracklets: [num_frames, max_obj, [track_id, x, y, z, qw, qx, qy, qz]]
        # frame_timestamps: [num_frames]
        super().__init__()
        tracklets = torch.from_numpy(tracklets).float().cuda()
        self.track_ids = tracklets[..., 0] # [num_frames, max_obj]
        self.input_trans = tracklets[..., 1:4] # [num_frames, max_obj, [x, y, z]]
        self.input_rots = tracklets[..., 4:8] # [num_frames, max_obj, [qw, qx, qy, qz]]
        self.timestamps = tracklet_timestamps
        self.camera_timestamps = camera_timestamps

        self.opt_track = cfg.model.nsg.opt_track
        if self.opt_track:
            self.opt_trans = nn.Parameter(torch.zeros_like(self.input_trans)).requires_grad_(True) 
            # [num_frames, max_obj, [dx, dy, dz]]

            self.opt_rots = nn.Parameter(torch.zeros_like(self.input_rots[..., :1])).requires_grad_(True) 
            # [num_frames, max_obj, [dtheta]
        
        self.obj_info = obj_info
        for track_id in self.obj_info.keys():
            self.obj_info[track_id]['track_idx'] = torch.argwhere(self.track_ids == track_id)

        self._build_batched_lookup()

    def _build_batched_lookup(self):
        """Pre-compute dense (T, L) lookup tables so per-track-id closest-frame
        queries can be batched without CPU syncs. Used by the *_batched APIs."""
        ts_arr = np.asarray(self.timestamps, dtype=np.float64)

        track_ids_sorted = sorted(int(t) for t in self.obj_info.keys())
        self._track_id_to_pos = {tid: i for i, tid in enumerate(track_ids_sorted)}

        rows_frame, rows_col, rows_ts = [], [], []
        max_len = 0
        for tid in track_ids_sorted:
            idx = self.obj_info[tid]['track_idx']  # (K, 2) cuda long
            frames = idx[:, 0].detach().cpu().numpy().astype(np.int64)
            cols = idx[:, 1].detach().cpu().numpy().astype(np.int64)
            ts = ts_arr[frames]
            order = np.argsort(ts)
            rows_frame.append(frames[order])
            rows_col.append(cols[order])
            rows_ts.append(ts[order])
            max_len = max(max_len, len(ts))

        T = len(track_ids_sorted)
        dense_frame = np.zeros((T, max_len), dtype=np.int64)
        dense_col = np.zeros((T, max_len), dtype=np.int64)
        dense_ts = np.full((T, max_len), np.inf, dtype=np.float64)
        for r in range(T):
            L = len(rows_ts[r])
            dense_frame[r, :L] = rows_frame[r]
            dense_col[r, :L] = rows_col[r]
            dense_ts[r, :L] = rows_ts[r]

        device = self.input_trans.device
        self.register_buffer('_dense_frame', torch.from_numpy(dense_frame).to(device), persistent=False)
        self.register_buffer('_dense_col', torch.from_numpy(dense_col).to(device), persistent=False)
        self.register_buffer('_dense_ts', torch.from_numpy(dense_ts).to(device), persistent=False)

    def _batched_closest_indices(self, track_ids, timestamp):
        """For each track_id in `track_ids`, find the two closest-in-time
        (frame, col) entries to `timestamp`. Fully on-device, no CPU sync.

        Returns f1, c1, t1, f2, c2, t2 each shape (B,)."""
        pos = torch.as_tensor(
            [self._track_id_to_pos[int(t)] for t in track_ids],
            dtype=torch.long, device=self._dense_ts.device,
        )
        ts_rows = self._dense_ts.index_select(0, pos)       # (B, L)
        fr_rows = self._dense_frame.index_select(0, pos)    # (B, L)
        co_rows = self._dense_col.index_select(0, pos)      # (B, L)

        dt = (ts_rows - float(timestamp)).abs()
        _, top2 = dt.topk(2, dim=1, largest=False)          # (B, 2)
        i1 = top2[:, :1]
        i2 = top2[:, 1:]
        f1 = fr_rows.gather(1, i1).squeeze(1)
        f2 = fr_rows.gather(1, i2).squeeze(1)
        c1 = co_rows.gather(1, i1).squeeze(1)
        c2 = co_rows.gather(1, i2).squeeze(1)
        t1 = ts_rows.gather(1, i1).squeeze(1)
        t2 = ts_rows.gather(1, i2).squeeze(1)
        return f1, c1, t1, f2, c2, t2

    @staticmethod
    def _slerp_batched(q0, q1, t):
        """Batched slerp matching the behavior of the per-id `quaternion_slerp`
        path (which delegates to roma.utils.unitquat_slerp — that function takes
        the shortest-arc path between q0 and q1)."""
        q0 = torch.nn.functional.normalize(q0, dim=-1)
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        q1 = torch.where(dot < 0, -q1, q1)
        # clamp strictly below 1.0: d/dx acos(x) = -1/sqrt(1-x^2) is -inf at
        # x=1, so even though the small-omega forward branch masks out the
        # s0/denom path, the chain rule still evaluates `0 * (-inf) = NaN`
        # through acos and poisons opt_rots.grad. For an actor with nearly
        # constant rotation between two adjacent frames (parked or
        # straight-driving car), q0 ≈ q1 ⇒ dot ≈ 1 ⇒ this triggers on iter 0
        # and silently NaNs every actor's _xyz by iter 1. (1 - 1e-7) keeps
        # the slerp result bit-identical for any rotation > ~0.025° while
        # making the acos gradient finite everywhere.
        cos_omega = dot.abs().clamp(max=1.0 - 1e-7)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega)
        t = t.to(q0.dtype).unsqueeze(-1)
        s0 = torch.sin((1.0 - t) * omega)
        s1 = torch.sin(t * omega)
        small = sin_omega.abs() < 1e-6
        denom = torch.where(small, torch.ones_like(sin_omega), sin_omega)
        w0 = torch.where(small, 1.0 - t, s0 / denom)
        w1 = torch.where(small, t, s1 / denom)
        return w0 * q0 + w1 * q1

    def _get_tracking_translation_batched(self, track_ids, timestamp):
        f1, c1, t1, f2, c2, t2 = self._batched_closest_indices(track_ids, timestamp)
        trans1 = self.input_trans[f1, c1]
        trans2 = self.input_trans[f2, c2]
        if self.opt_track:
            trans1 = trans1 + self.opt_trans[f1, c1]
            trans2 = trans2 + self.opt_trans[f2, c2]
        dtype = trans1.dtype
        w1 = (t2 - float(timestamp)).to(dtype).unsqueeze(-1)
        w2 = (float(timestamp) - t1).to(dtype).unsqueeze(-1)
        denom = (t2 - t1).to(dtype).unsqueeze(-1)
        return (trans1 * w1 + trans2 * w2) / denom

    def _get_tracking_rotation_batched(self, track_ids, timestamp):
        f1, c1, t1, f2, c2, t2 = self._batched_closest_indices(track_ids, timestamp)
        rots1 = self.input_rots[f1, c1]
        rots2 = self.input_rots[f2, c2]
        if self.opt_track:
            # opt_rots has a trailing length-1 dim; squeeze so broadcasting matches
            # quaternion_raw_multiply_theta's per-element shape contract (B,) vs (B,4).
            theta1 = self.opt_rots[f1, c1].squeeze(-1)
            # NOTE: mirrors the (likely buggy) per-id path which reused f1/rots1
            # for the second sample. Preserved here to keep training behavior
            # bit-identical; revisit separately.
            theta2 = self.opt_rots[f1, c2].squeeze(-1)
            rots1 = quaternion_raw_multiply_theta(rots1, theta1)
            rots2 = quaternion_raw_multiply_theta(rots1, theta2)
        r = (float(timestamp) - t1) / (t2 - t1)
        return self._slerp_batched(rots1, rots2, r)

    def get_tracking_translation_batched(self, track_ids, camera: Camera):
        """Batched equivalent of get_tracking_translation for a list of track_ids.
        Returns (B, 3) tensor in graph_obj_list order."""
        if len(track_ids) == 0:
            return torch.empty(0, 3, device=self._dense_ts.device)
        if self.opt_track and camera.meta['is_val']:
            # Val branch needs per-track camera-time clamping; fall back to loop.
            outs = [self.get_tracking_translation(int(t), camera) for t in track_ids]
            return torch.stack(outs, dim=0)
        return self._get_tracking_translation_batched(track_ids, camera.meta['timestamp'])

    def get_tracking_rotation_batched(self, track_ids, camera: Camera):
        """Batched equivalent of get_tracking_rotation for a list of track_ids.
        Returns (B, 4) tensor in graph_obj_list order."""
        if len(track_ids) == 0:
            return torch.empty(0, 4, device=self._dense_ts.device)
        if self.opt_track and camera.meta['is_val']:
            # Per-id get_tracking_rotation_ returns (1, 4) when opt_track is on,
            # because opt_rots has a trailing length-1 dim that broadcasts up
            # through quaternion_raw_multiply_theta. Reshape to (4,) before
            # stacking so this branch matches the (B, 4) contract of the
            # non-val path.
            outs = [self.get_tracking_rotation(int(t), camera).reshape(4) for t in track_ids]
            return torch.stack(outs, dim=0)
        return self._get_tracking_rotation_batched(track_ids, camera.meta['timestamp'])

    def save_state_dict(self, is_final):
        state_dict = dict()
        if self.opt_track:
            state_dict['params'] = self.state_dict()
        if not is_final:
            state_dict['optimizer'] = self.optimizer.state_dict()
        return state_dict
        
    def load_state_dict(self, state_dict):
        if self.opt_track:
            super().load_state_dict(state_dict['params'])
            if cfg.mode == 'train' and 'optimizer' in state_dict:
                self.optimizer.load_state_dict(state_dict['optimizer'])

    def training_setup(self):
        args = cfg.optim
        if self.opt_track:
            params = [
                {'params': [self.opt_trans], 'lr': args.track_position_lr_init, 'name': 'opt_trans'},
                {'params': [self.opt_rots], 'lr': args.track_rotation_lr_init, 'name': 'opt_rots'},
            ]
            
            self.opt_trans_scheduler_args = get_expon_lr_func(lr_init=args.track_position_lr_init,
                                                    lr_final=args.track_position_lr_final,
                                                    lr_delay_mult=args.track_position_lr_delay_mult,
                                                    max_steps=args.track_position_max_steps,
                                                    warmup_steps=args.opacity_reset_interval)
            
            self.opt_rots_scheduler_args = get_expon_lr_func(lr_init=args.track_rotation_lr_init,
                                                    lr_final=args.track_rotation_lr_final,
                                                    lr_delay_mult=args.track_rotation_lr_delay_mult,
                                                    max_steps=args.track_rotation_max_steps,
                                                    warmup_steps=args.opacity_reset_interval)    
            
            self.optimizer = torch.optim.Adam(params=params, lr=0, eps=1e-15)
    
    def update_learning_rate(self, iteration):
        if self.opt_track:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "opt_trans":
                    lr = self.opt_trans_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "opt_rots":
                    lr = self.opt_rots_scheduler_args(iteration)
                    param_group['lr'] = lr
        
    def update_optimizer(self, scaler=None):
        if self.opt_track:
            if scaler is not None:
                # Skip if no gradients exist (e.g. hard_depth loss doesn't flow here)
                has_grads = any(
                    p.grad is not None
                    for group in self.optimizer.param_groups
                    for p in group["params"]
                )
                if has_grads:
                    scaler.unscale_(self.optimizer)
                    scaler.step(self.optimizer)
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        
    def find_closest_indices(self, track_id, timestamp):
        track_idx = self.obj_info[track_id]['track_idx']
        frame_idx = track_idx[:, 0].cpu()
        frame_timestamps = np.array(self.timestamps[frame_idx])
        assert len(frame_timestamps) > 1
        delta_timestamps = np.abs(frame_timestamps - timestamp)
        idx1, idx2 = np.argsort(delta_timestamps)[:2]
        return track_idx[idx1], track_idx[idx2]

    
    def find_closest_camera_timestamps(self, track_id, camera: Camera):
        timestamp = camera.meta['timestamp']
        cam = camera.meta['cam']
        camera_timestamps = self.camera_timestamps[cam]['train_timestamps']
        start_timestamp = self.obj_info[track_id]['start_timestamp']
        end_timestamp = self.obj_info[track_id]['end_timestamp']
        camera_timestamps = np.array([x for x in camera_timestamps if x >= start_timestamp and x <= end_timestamp])
        if len(camera_timestamps) < 2:
            return None, None
        else:
            delta_timestamps = np.abs(camera_timestamps - timestamp)
            idx1, idx2 = np.argsort(delta_timestamps)[:2]            
            return camera_timestamps[idx1], camera_timestamps[idx2]
            
    def get_tracking_translation_(self, track_id, timestamp):
        ind1, ind2 = self.find_closest_indices(track_id, timestamp)
        frame_ind1, frame_ind2 = ind1[0], ind2[0]
        column_ind1, column_ind2 = ind1[1], ind2[1]
        timestamp1, timestamp2 = self.timestamps[frame_ind1.cpu()], self.timestamps[frame_ind2.cpu()]

        if self.opt_track:
            trans1 = self.input_trans[frame_ind1, column_ind1] + self.opt_trans[frame_ind1, column_ind1]
            trans2 = self.input_trans[frame_ind2, column_ind2] + self.opt_trans[frame_ind2, column_ind2]
        else:
            trans1 = self.input_trans[frame_ind1, column_ind1]
            trans2 = self.input_trans[frame_ind2, column_ind2]
        
        trans = (trans1 * (timestamp2 - timestamp) + trans2 * (timestamp - timestamp1)) / (timestamp2 - timestamp1)        
            
        return trans
    
    def get_tracking_translation(self, track_id, camera: Camera):
        if self.opt_track and camera.meta['is_val']:
            timestamp1, timestamp2 = self.find_closest_camera_timestamps(track_id, camera)
            if timestamp1 is None:
                return self.get_tracking_translation_(track_id, camera.meta['timestamp'])
            else:
                timestamp = camera.meta['timestamp']
                trans1 = self.get_tracking_translation_(track_id, timestamp1)
                trans2 = self.get_tracking_translation_(track_id, timestamp2)
                trans = (trans1 * (timestamp2 - timestamp) + trans2 * (timestamp - timestamp1)) / (timestamp2 - timestamp1)
                return trans
        else:
            return self.get_tracking_translation_(track_id, camera.meta['timestamp'])

    def get_tracking_rotation_(self, track_id, timestamp):
        ind1, ind2 = self.find_closest_indices(track_id, timestamp)
        frame_ind1, frame_ind2 = ind1[0], ind2[0]
        column_ind1, column_ind2 = ind1[1], ind2[1]
        timestamp1, timestamp2 = self.timestamps[frame_ind1.cpu()], self.timestamps[frame_ind2.cpu()]
        
        if self.opt_track:
            rots1 = self.input_rots[frame_ind1, column_ind1]
            rots2 = self.input_rots[frame_ind2, column_ind2]
            theta1 = self.opt_rots[frame_ind1, column_ind1]
            theta2 = self.opt_rots[frame_ind1, column_ind2]
            rots1 = quaternion_raw_multiply_theta(rots1, theta1)
            rots2 = quaternion_raw_multiply_theta(rots1, theta2)
        else:
            rots1 = self.input_rots[frame_ind1, column_ind1]
            rots2 = self.input_rots[frame_ind2, column_ind2]

        r = (timestamp - timestamp1) / (timestamp2 - timestamp1)
        rots = quaternion_slerp(rots1, rots2, r)

        return rots
   
    def get_tracking_rotation(self, track_id, camera: Camera):
        if self.opt_track and camera.meta['is_val']:
            timestamp1, timestamp2 = self.find_closest_camera_timestamps(track_id, camera)
            if timestamp1 is None:
                return self.get_tracking_rotation_(track_id, camera.meta['timestamp'])
            else:
                timestamp = camera.meta['timestamp']
                rots1 = self.get_tracking_rotation_(track_id, timestamp1)
                rots2 = self.get_tracking_rotation_(track_id, timestamp2)
                r = (timestamp - timestamp1) / (timestamp2 - timestamp1)
                rots = quaternion_slerp(rots1, rots2, r)
                return rots
        else:
            return self.get_tracking_rotation_(track_id, camera.meta['timestamp'])