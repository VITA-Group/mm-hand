import sys
import cv2
from pytorch_ssim import ssim
import torch
from inception import inception_v3
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from scipy.stats import entropy
from torchvision import transforms
from hpe_estimator import HPEstimator
class Evaluator:
    def __init__(self, model:torch.nn.Module, weights:str, w_cpm2d:str, w_cpm3d:str, device:int):
        print("initialize Evaluator")
        self.model = model
        self.device = 'cpu' if device < 0 else f"cuda:{device}"
        print(f"device : {self.device}")
        self.model.load_state_dict(self._get_weights(weights))
        print("model's weights loaded")
        self.inception_model = inception_v3(True, transform_input=False, init_weights=False)
        print("inception model initialized")
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(mode=None),
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if self.device != 'cpu':
            self.model = self.model.to(device)
            self.inception_model = self.inception_model.to(device)
        self.inception_model.eval()
        self.model.eval()
        self.IS_cache_zize = 64
        self.IS_cache = []
        self.IS_scores = []
        self.SSIM_scores = []
        self.PCK_scores = HPEstimator(w_cpm2d, w_cpm3d, self.device)

    def feed(self, xyz:torch.tensor, gt_image:torch.tensor):
        """
        xyz.shape = 1x21x3
        gt_image.shape = 1xCxWxH
        """
        with torch.no_grad():
            input_image = self._create_input_image(xyz.squeeze(0))
            pred_image = self.model(input_image)
        if len(gt_image.shape) < 4 :
            gt_image = gt_image.unsqueeze(dim=0)
        self._feed(pred_image, gt_image, xyz)

    def _feed(self, pred_image : torch.tensor, gt_image :torch.tensor, xyz: torch.tensor = None):
        """
        pred_image.shape (1xCxHxW)
        gt_image.shape (1xCxHxW)
        xyz.shape(1x21x3)
        """
        assert len(pred_image.shape) == 4 and len(gt_image.shape) == 4
        self._get_IS_score(pred_image)
        self._get_SSIM_score(pred_image, gt_image)
        self._get_pck_score(pred_image, xyz)

    def evaluate(self):
        if len(self.IS_cache):
            self._get_IS_score(None, force=True)
        scores = {}
        scores['IS_avg'] = np.mean(self.IS_scores)
        scores['IS_std'] = np.std(self.IS_scores)
        scores['SSIM_avg'] = np.mean(self.SSIM_scores)
        scores['SSIM_std'] = np.std(self.SSIM_scores)
        pck2d, pck3d =  self.PCK_scores.get_results(30, 20)
        scores['pck2d_auc'] = pck2d[2]
        scores['pck3d_auc'] = pck3d[2]

        return scores

    def _get_pck_score(self, pred_image, xyz):
        self.PCK_scores.feed(pred_image.squeeze(0), xyz.squeeze(0))


    def _get_IS_score(self, pred_image: torch.tensor, force=False):
        """Computes the inception score of the generated images imgs

        imgs -- Torch dataset of (1xCxHxW) numpy images normalized in the range [-1, 1]
        cuda -- whether or not to run on GPU
        batch_size -- batch size for feeding into Inception v3
        splits -- number of splits
        """

        if (len(self.IS_cache) and len(self.IS_cache) % self.IS_cache_zize==0) or force:
            mean, std = inception_score(self.IS_cache, self.inception_model, splits=1)
            self.IS_scores.append(mean)
            self.IS_cache = []

        if not force:
            pred_image = self.preprocess(pred_image.squeeze(dim=0).cpu())
            pred_image = pred_image.to(self.device) if self.device != 'cpu' else pred_image
            self.IS_cache.append(pred_image)

    def _get_SSIM_score(self, pred_image:torch.tensor, gt_image:torch.tensor):
        """
        pred_image.shape == (1xCxWxH)
        gt_image.shape == (1xCxWxH)

        """
        gt_image = gt_image / 255
        if self.device != 'cpu':
            pred_image = pred_image.to(self.device)
            gt_image = gt_image.to(self.device)
        score = ssim(pred_image, gt_image)
        self.SSIM_scores.append(score.cpu().numpy())

    def _create_input_image(self, xyz):
      # assuming xyz = (21, 3)
      assert(xyz.shape == (21, 3))
      uv = xyz[:, 0:2]
      z = xyz[:, -1]
      width, height = 256, 256
      numpy_image = generate_jointsmap(uv.cpu().numpy(), z.cpu().numpy(), width, height)
      numpy_image = cv2.normalize(numpy_image, None, alpha=1, norm_type=cv2.NORM_MINMAX)
      input_tensor = torch.tensor(numpy_image).permute(2,0,1).type(torch.float32)
      input_tensor = input_tensor.unsqueeze(0)
      input_tensor = input_tensor.to(self.device)
      return input_tensor

    def _get_weights(self, path):
        _weights = torch.load(path, map_location = self.device)
        weights = {}
        for k, v in _weights.items():
            if "module" in k :
                k = k.split(".")
                k = ".".join(k[1::])
                weights[k] = v
            else:
                weights[k] = v
        return weights

    def clean(self):
        self.IS_scores = []
        self.SSIM_scores = []

def generate_jointsmap(uv_coord, depth, width, height, channel=3):
    canvas = np.ones((height, width, channel)) * sys.maxsize
    _canvas = canvas.copy()
    bones = [

        ((0, 17), [160] * channel),
        ((0, 1), [170] * channel),
        ((0, 5), [180] * channel),
        ((0, 9), [190] * channel),
        ((0, 13), [200] * channel),

        ((17, 18), [130] * channel),
        ((18, 19), [140] * channel),
        ((19, 20), [150] * channel),

        ((1, 2), [10] * channel),
        ((2, 3), [20] * channel),
        ((3, 4), [30] * channel),

        ((5, 6), [40] * channel),
        ((6, 7), [50] * channel),
        ((7, 8), [60] * channel),

        ((9, 10), [70] * channel),
        ((10, 11), [80] * channel),
        ((11, 12), [90] * channel),

        ((13, 14), [100] * channel),
        ((14, 15), [110] * channel),
        ((15, 16), [120] * channel),
    ]

    for connection, color in bones:
        temp_canvas = np.ones(canvas.shape) * sys.maxsize

        coord1 = uv_coord[connection[0]]
        coord2 = uv_coord[connection[1]]

        coords = np.stack([coord1, coord2])
        avg_depth = (depth[connection[0]] + depth[connection[1]]) / 2
        x = coords[:, 0]
        y = coords[:, 1]
        mX = x.mean()
        mY = y.mean()
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = np.math.degrees(np.math.atan2(y[0] - y[1], x[0] - x[1]))
        radius = 5
        polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), radius), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(temp_canvas, polygon, [avg_depth] * channel)
        _canvas = np.minimum(_canvas, temp_canvas)
        canvas[_canvas==avg_depth] = color[0]
    canvas[canvas==sys.maxsize] = 0
    return canvas

def inception_score(imgs, inception_model, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)
    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    def get_pred(x):
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    return np.mean(split_scores), np.std(split_scores)
