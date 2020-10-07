from networks.net_hpm2d import Hpm2d
from networks.net_hpm3d import Hpm3d
import torch
import numpy as np

BONES = [
        ((0, 17), [160] ),
        ((0, 1), [170] ),
        ((0, 5), [180] ),
        ((0, 9), [190] ),
        ((0, 13), [200] ),

        ((17, 18), [130] ),
        ((18, 19), [140] ),
        ((19, 20), [150] ),

        ((1, 2), [10] ),
        ((2, 3), [20] ),
        ((3, 4), [30] ),

        ((5, 6), [40] ),
        ((6, 7), [50] ),
        ((7, 8), [60] ),

        ((9, 10), [70] ),
        ((10, 11), [80] ),
        ((11, 12), [90] ),

        ((13, 14), [100] ),
        ((14, 15), [110] ),
        ((15, 16), [120] ),
    ]
class EvalUtil:
    """ Util class for evaluation networks.
    """
    def __init__(self, num_kp=21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        self._avg_length = []
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, keypoint_gt, keypoint_vis, keypoint_pred):
        """ Used to feed data to the class. Stores the euclidean distance between gt and pred, when it is visible. """
        keypoint_gt = np.squeeze(keypoint_gt)
        keypoint_pred = np.squeeze(keypoint_pred)
        keypoint_vis = np.squeeze(keypoint_vis).astype('bool')

        assert len(keypoint_gt.shape) == 2
        assert len(keypoint_pred.shape) == 2
        assert len(keypoint_vis.shape) == 1

        # calc euclidean distance
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))

        num_kp = keypoint_gt.shape[0]
        for i in range(num_kp):
            if keypoint_vis[i]:
                self.data[i].append(euclidean_dist[i])
        self._avg_length.append(self._get_avg_length(keypoint_gt))

    def _get_avg_length(self, kp):
        ans = []
        for connection, _ in BONES:
            coord1 = kp[connection[0]]
            coord2 = kp[connection[1]]
            ans.append(np.linalg.norm(coord1-coord2))
        return np.mean(ans)

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype('float'))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.arange(0, np.mean(self._avg_length) * 1/3)
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)

        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints
        return epe_mean_all, epe_median_all, auc_all, pck_curve_all, thresholds

class HPEstimator:
    def __init__(self, weights2d, weights3d, device):
        self.hpm2d = Hpm2d(21, 3, False)
        self.hpm3d = Hpm3d(21, 21)
        self.hpm2d.load_state_dict(self._get_weights(weights2d))
        self.hpm3d.load_state_dict(self._get_weights(weights3d))
        self.hpm2d= self.hpm2d.to(device)
        self.hpm3d= self.hpm3d.to(device)
        self.device = device
        self.hpm2d.eval()
        self.hpm3d.eval()

        self.eval2d = EvalUtil(21)
        self.eval3d = EvalUtil(21)

    def feed(self, image, gt):
        """Assuming image to be a tensor object with dimension of (3x256x256)
        and gt to be a tensor object of dimension (21 x 3) with real depth value
        """
        assert(image.shape == (3, 256, 256))
        assert(gt.shape == (21, 3))
        image = torch.unsqueeze(image, dim=0)
        gt = gt.cpu().numpy()
        # chaning real world unit to pixel units

        gt[:, -1] = (gt[:, -1]/700.) * 256.
        with torch.no_grad():
            predict_2d = self.hpm2d(image.to(self.device))[-1]
            predict_3d = self.hpm3d(predict_2d)[-1]

            KP, H, W = 21, 256, 256
            predict_2d = predict_2d.cpu().view(KP, -1)
            predict_3d = predict_3d.cpu()
            p2d, p3d = [], []
            for heatmap, z in zip(predict_2d, predict_3d):
                index = torch.max(heatmap, 0)[-1]
                y, x = index / H, index % W
                p2d.append([x.cpu().numpy(), y.cpu().numpy()])
                p3d.append([x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy()*H])

            self.eval2d.feed(gt[:, 0:2], np.ones(21), p2d)
            self.eval3d.feed(gt, np.ones(21), p3d)

    def get_results(self, pixel_offset, n_steps):
        results2d = self.eval2d.get_measures(0, pixel_offset, n_steps)
        results3d = self.eval3d.get_measures(0, pixel_offset, n_steps)
        return results2d, results3d

    def _get_weights(self, path):
        weights = torch.load(path)
        # get rid of "modules param
        state_dict = {}
        for k, v in weights.items():
            k = k.split('.')
            if 'module' == k[0]:
                k = '.'.join(k[1::])
            else:
                k = '.'.join(k)
            state_dict[k] = v
        return state_dict

if __name__ == "__main__":
    from data.STB_dataset import STBdataset
    from data.RHD_dataset import RHDdataset
    from easydict import EasyDict as edict
    from tqdm import tqdm
    opt = edict()
    opt.dataroot = "./datasets/rhd_dataset/test"
    opt.isTrain = False
    dataset = RHDdataset(opt)
    w2d = '../../90_net_Hpm.pth'
    w3d = './hpm3d_latest/latest_net_Hpm3d.pth'
    est = HPEstimator(w2d, w3d, 0)
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        est.feed(sample['A'], sample['xyz'])
    result2d, result3d = est.get_results(30, 20)
    auc2d = result2d[2]
    auc3d = result3d[2]
    print(f"auc 2d: {auc2d}")
    print(f"auc 3d: {auc3d}")

