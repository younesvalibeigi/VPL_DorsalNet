"""
Dataset helpers for the ANN readout experiment.

Used in the paper's pipeline:
    add_bw_gaussian_circle        - grayscale Gaussian noise inside a circle,
                                    added on every training/evaluation pass
    move_center_square_to_corner  - builds the location-transfer test set
"""

import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageSequence


class StimulusDatasetUtils:
    """Helper utilities for:
    - Loading generated GIF stimuli as tensors
    - Building behavioural datasets (same/different, binary classification)
    - Data augmentation (Gaussian noise, rotation, mirroring)
    - Normalization
    - Circular masking
    - Moving the central patch to another position (location test)
    - Averaging helpers (numpy and torch)
    """

    def __init__(self, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.to_tensor = transforms.ToTensor()

    # ------------------------------------------------------------------
    # Load generated GIF stimulus sequences into tensors
    # (alternative to the raw tensors; not used when var.raw_data = True)
    # ------------------------------------------------------------------
    def recall_generated_data(self, var, stim_address, task="same-different", layer_stim_test=None, device=None):
        num_cond = var.num_cond

        levels = range(0, 25 + 1)  # all 26 saved levels

        basic_anims = range(2)
        stimuli = [[None for _ in basic_anims] for _ in levels]
        filename = f"level_{0}_basic_animation_{0}.gif"
        frames = self._load_gif(os.path.join(stim_address, filename))
        C = frames.shape[-4]
        F = frames.shape[-3]
        H = frames.shape[-2]
        W = frames.shape[-1]

        if task == "same-different":
            beh_X = torch.zeros((len(levels) * num_cond, 2, C, F, H, W))
        elif task == 'binary':
            beh_X = torch.zeros((len(levels) * num_cond, C, F, H, W))

        beh_Y = torch.zeros((len(levels) * num_cond, 1))

        for li, level in enumerate(levels):
            for a in basic_anims:
                filename = f"level_{level}_basic_animation_{a}.gif"
                frames = self._load_gif(os.path.join(stim_address, filename))
                stimuli[li][a] = frames.unsqueeze(0)  # [1, 3, 9, H, W]

            if task == "same-different":
                if num_cond == 4:
                    beh_X[li * num_cond] = torch.cat((stimuli[li][0], stimuli[li][1]), dim=0)
                    beh_Y[li * num_cond] = 0
                    beh_X[li * num_cond + 1] = torch.cat((stimuli[li][1], stimuli[li][0]), dim=0)
                    beh_Y[li * num_cond + 1] = 0
                    beh_X[li * num_cond + 2] = torch.cat((stimuli[li][0], stimuli[li][0]), dim=0)
                    beh_Y[li * num_cond + 2] = 1
                    beh_X[li * num_cond + 3] = torch.cat((stimuli[li][1], stimuli[li][1]), dim=0)
                    beh_Y[li * num_cond + 3] = 1

                elif num_cond == 2:
                    beh_X[li * num_cond] = torch.cat((stimuli[li][0], stimuli[li][1]), dim=0)
                    beh_Y[li * num_cond] = 0
                    beh_X[li * num_cond + 1] = torch.cat((stimuli[li][0], stimuli[li][0]), dim=0)
                    beh_Y[li * num_cond + 1] = 1

            elif task == 'binary':
                beh_X[li * num_cond] = stimuli[li][0]
                beh_Y[li * num_cond] = 0
                beh_X[li * num_cond + 1] = stimuli[li][1]
                beh_Y[li * num_cond + 1] = 1

        beh_X = beh_X.to(device)
        beh_Y = beh_Y.to(device)

        return beh_X, beh_Y

    def _load_gif(self, filepath):
        frames = []
        with Image.open(filepath) as img:
            for frame in ImageSequence.Iterator(img):
                t = self.to_tensor(frame)
                if t.shape[0] == 1:
                    t = t.repeat(3, 1, 1)
                frames.append(t)
        tensor = torch.stack(frames, dim=1)
        return tensor  # [3, T, H, W]

    # ------------------------------------------------------------------
    # Gaussian noise utilities
    # ------------------------------------------------------------------
    def add_gaussian_noise(self, tensor, mean=0.0, std=1.0):
        noise = torch.normal(mean=mean, std=std, size=tensor.size(), device=tensor.device)
        return tensor + noise

    def expand_with_noise(self, beh_X, beh_Y, num_augment=2, std=0.0):
        aug = [self.add_gaussian_noise(beh_X, std=std) for _ in range(num_augment)]
        combined_X = torch.cat([beh_X] + aug, dim=0)
        combined_Y = beh_Y.repeat(num_augment + 1, 1)
        return combined_X, combined_Y

    # ------------------------------------------------------------------
    # Rotations & mirrored variants
    # ------------------------------------------------------------------
    def rotate_90(self, tensor):
        return torch.rot90(tensor, k=1, dims=[-2, -1])

    def rotate_180(self, tensor):
        return tensor.flip(-2, -1)

    def mirror(self, tensor):
        return tensor.flip(-1)

    # ------------------------------------------------------------------
    # Averaging helpers
    # ------------------------------------------------------------------
    def average_every_n(self, lst, n):
        return [sum(lst[i:i + n]) / n for i in range(0, len(lst), n)]

    def average_same_different(self, lst, n):
        nn = n // 2
        diff = [sum(lst[i:i + nn]) / nn for i in range(0, len(lst), n)]
        same = [sum(lst[i + nn:i + 2 * nn]) / nn for i in range(0, len(lst), n)]
        return same, diff

    # ------------------------------------------------------------------
    # Normalize spatial frames
    # ------------------------------------------------------------------
    def normalize_data(self, beh_X):
        mean = beh_X.mean(dim=(-2, -1), keepdim=True)
        std = beh_X.std(dim=(-2, -1), keepdim=True)
        std[std == 0] = 1
        return (beh_X - mean) / std

    # ------------------------------------------------------------------
    # Flatten paired dataset: [pair0_A, pair0_B, pair1_A, pair1_B, ...]
    # ------------------------------------------------------------------
    def combine_input_pairs(self, beh_X):
        return torch.cat([beh_X[i] for i in range(beh_X.shape[0])], dim=0)

    # ------------------------------------------------------------------
    # Circular noise patch (Section 2.4.2)
    # ------------------------------------------------------------------
    def add_bw_gaussian_circle(self, data, mean=0., std=0.1, radius=None):
        """Add grayscale (same value in R, G and B) Gaussian noise inside a
        centred circle of `radius` pixels, then clamp the result to [0, 1].

        Note: because of the final clamp, pixel values below 0 become 0 and
        values above 1 become 1.
        """
        B, C, T, H, W = data.shape
        noise = torch.randn((B, 1, T, H, W), device=self.device) * std + mean
        noise = noise.expand(-1, 3, -1, -1, -1)

        yy, xx = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing="ij"
        )
        center = (H // 2, W // 2)
        if radius is None:
            radius = min(H, W) // 2

        mask = ((yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius ** 2).float()
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        mask = mask.expand(B, 3, T, -1, -1)

        noisy = data + noise * mask
        return torch.clamp(noisy, 0, 1)

    # ------------------------------------------------------------------
    # Circular mask (gray background)
    # ------------------------------------------------------------------
    def apply_circular_mask(self, video_tensor, radius=50, gray_value=0.5, center=None):
        B, C, T, H, W = video_tensor.shape
        cy = H // 2 if center is None else center[0]
        cx = W // 2 if center is None else center[1]

        yy, xx = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing="ij"
        )
        mask = ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2).float()
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        return video_tensor * mask + gray_value * (1 - mask)

    # ------------------------------------------------------------------
    # Move the central patch (location-transfer test, Section 2.4.3)
    # ------------------------------------------------------------------
    def move_center_square_to_corner(
            self,
            video_tensor,
            edge=100,
            gray_value=0.5,
            position='top-right',
            output_size=None,
            shift_px=None,
    ):
        """
        Extract the central `edge` x `edge` square and paste it flush into a
        corner of a uniform canvas, or shift it from the centre toward that
        corner by `shift_px` pixels.

        With edge = 2 * 66 = 132 on a 164 x 164 canvas and
        position = 'bottom-right', the stimulus moves 16 px right and 16 px
        down, so it still partly overlaps the training location.

        Args:
            video_tensor (torch.Tensor): (B, C, T, H, W)
            edge (int): size of the square
            gray_value (float): background fill value
            position (str): 'top-right', 'top-left', 'bottom-right', 'bottom-left'
            output_size (int or tuple, optional): output (H, W)
            shift_px (int or None): pixels to shift from the centre toward the
                corner. None or 0 places the square flush in the corner.
        Returns:
            torch.Tensor: (B, C, T, H_out, W_out)
        """
        B, C, T, H, W = video_tensor.shape
        device = video_tensor.device

        if output_size is None:
            H_out, W_out = H, W
        elif isinstance(output_size, int):
            H_out = W_out = output_size
        else:
            H_out, W_out = output_size

        # Source (central square)
        half = edge // 2
        cy, cx = H // 2, W // 2
        y1_src, y2_src = cy - half, cy + half
        x1_src, x2_src = cx - half, cx + half

        s = 0 if shift_px is None else int(shift_px)

        if s == 0:
            # Flush placement in the requested corner
            if position == 'top-right':
                y1_dst, y2_dst = 0, edge
                x1_dst, x2_dst = W_out - edge, W_out
            elif position == 'top-left':
                y1_dst, y2_dst = 0, edge
                x1_dst, x2_dst = 0, edge
            elif position == 'bottom-right':
                y1_dst, y2_dst = H_out - edge, H_out
                x1_dst, x2_dst = W_out - edge, W_out
            elif position == 'bottom-left':
                y1_dst, y2_dst = H_out - edge, H_out
                x1_dst, x2_dst = 0, edge
            else:
                raise ValueError("Invalid position. Choose from 'top-right', 'top-left', 'bottom-right', 'bottom-left'.")
        else:
            # Move from the centre toward the chosen corner by s pixels
            dy = -s if "top" in position else +s
            dx = -s if "left" in position else +s

            cy_out, cx_out = H_out // 2, W_out // 2
            y1_dst = (cy_out - half) + dy
            x1_dst = (cx_out - half) + dx
            y2_dst = y1_dst + edge
            x2_dst = x1_dst + edge

            # Clamp to stay inside the canvas
            y1_dst = max(0, min(y1_dst, H_out - edge))
            x1_dst = max(0, min(x1_dst, W_out - edge))
            y2_dst = y1_dst + edge
            x2_dst = x1_dst + edge

        output = torch.full((B, C, T, H_out, W_out), gray_value, device=device, dtype=video_tensor.dtype)
        output[:, :, :, y1_dst:y2_dst, x1_dst:x2_dst] = video_tensor[:, :, :, y1_src:y2_src, x1_src:x2_src]

        return output

    def move_center_square_to_corner2(
            self,
            video_tensor,
            edge=100,
            gray_value=0.5,
            position='top-right',
            output_size=None,
            shift_px=None,
    ):
        """
        Like move_center_square_to_corner, but also accepts the sides
        'top', 'bottom' (or 'down'), 'left', 'right'. Used only for the
        optional microsaccade simulation (not used in the paper).
        """
        B, C, T, H, W = video_tensor.shape
        device = video_tensor.device

        if output_size is None:
            H_out, W_out = H, W
        elif isinstance(output_size, int):
            H_out = W_out = output_size
        else:
            H_out, W_out = output_size

        half = edge // 2
        cy, cx = H // 2, W // 2
        y1_src, y2_src = cy - half, cy + half
        x1_src, x2_src = cx - half, cx + half

        pos = position.lower().strip()
        if pos == "down":
            pos = "bottom"

        s = 0 if shift_px is None else int(shift_px)

        if s == 0:
            if pos == 'top-right':
                y1_dst, y2_dst = 0, edge
                x1_dst, x2_dst = W_out - edge, W_out
            elif pos == 'top-left':
                y1_dst, y2_dst = 0, edge
                x1_dst, x2_dst = 0, edge
            elif pos == 'bottom-right':
                y1_dst, y2_dst = H_out - edge, H_out
                x1_dst, x2_dst = W_out - edge, W_out
            elif pos == 'bottom-left':
                y1_dst, y2_dst = H_out - edge, H_out
                x1_dst, x2_dst = 0, edge
            elif pos == 'top':
                y1_dst, y2_dst = 0, edge
                x1_dst = (W_out - edge) // 2
                x2_dst = x1_dst + edge
            elif pos == 'bottom':
                y1_dst, y2_dst = H_out - edge, H_out
                x1_dst = (W_out - edge) // 2
                x2_dst = x1_dst + edge
            elif pos == 'left':
                x1_dst, x2_dst = 0, edge
                y1_dst = (H_out - edge) // 2
                y2_dst = y1_dst + edge
            elif pos == 'right':
                x1_dst, x2_dst = W_out - edge, W_out
                y1_dst = (H_out - edge) // 2
                y2_dst = y1_dst + edge
            else:
                raise ValueError(
                    "Invalid position. Choose from "
                    "'top-right','top-left','bottom-right','bottom-left','top','bottom'/'down','left','right'."
                )
        else:
            dy = 0
            dx = 0

            if "top" in pos:
                dy = -s
            elif "bottom" in pos:
                dy = +s

            if "left" in pos:
                dx = -s
            elif "right" in pos:
                dx = +s

            if pos == "top":
                dy = -s; dx = 0
            elif pos == "bottom":
                dy = +s; dx = 0
            elif pos == "left":
                dx = -s; dy = 0
            elif pos == "right":
                dx = +s; dy = 0

            valid = {
                'top-right', 'top-left', 'bottom-right', 'bottom-left',
                'top', 'bottom', 'left', 'right'
            }
            if pos not in valid:
                raise ValueError(
                    "Invalid position. Choose from "
                    "'top-right','top-left','bottom-right','bottom-left','top','bottom'/'down','left','right'."
                )

            cy_out, cx_out = H_out // 2, W_out // 2
            y1_dst = (cy_out - half) + dy
            x1_dst = (cx_out - half) + dx
            y2_dst = y1_dst + edge
            x2_dst = x1_dst + edge

            y1_dst = max(0, min(y1_dst, H_out - edge))
            x1_dst = max(0, min(x1_dst, W_out - edge))
            y2_dst = y1_dst + edge
            x2_dst = x1_dst + edge

        output = torch.full(
            (B, C, T, H_out, W_out),
            gray_value,
            device=device,
            dtype=video_tensor.dtype
        )

        output[:, :, :, y1_dst:y2_dst, x1_dst:x2_dst] = video_tensor[:, :, :, y1_src:y2_src, x1_src:x2_src]

        return output

    # ------------------------------------------------------------------
    # Numpy helpers
    # ------------------------------------------------------------------
    def average_every_n_numpy(self, arr, n):
        assert arr.shape[-1] % n == 0
        new_shape = arr.shape[:-1] + (arr.shape[-1] // n, n)
        return arr.reshape(new_shape).mean(axis=-1)

    def mean_and_se(self, data):
        mean = 1 - np.mean(data, axis=0)
        se = np.std(data, axis=0, ddof=1) / np.sqrt(data.shape[0])
        return mean, se
