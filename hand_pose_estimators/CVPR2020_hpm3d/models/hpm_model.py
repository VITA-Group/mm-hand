"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pai4000ata_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input dataset and perform dataset pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from .base_model import BaseModel
from . import networks
import apex
from apex import amp
import numpy as np
# Hand Pose Module 2-D
class hpmModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument("--data_mode", type=str, default='hpm2d')
        parser.add_argument('--num_loss', type=int, default=2)
        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        # Hpm2d.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        super().__init__(opt)
        self.loss_names = ['mse', 'lz']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['ground_truth', 'l6', 'data_A']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        self.model_names = ['Hpm2d', 'Hpm3d']
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        optimizer = torch.optim.Adam if not self.opt.distributed else apex.optimizers.FusedAdam
        optimizers = [[optimizer, 'hpm2d'], [optimizer, 'hpm3d']]

        self.networkFactory = networks.NetworkInitializer(networks=['hpm2d', 'hpm3d'],
                                                    optimizers=optimizers, options=opt)

        # self.netHpm.cuda()
        if self.isTrain:  # only defined during training time
            self.criterionLoss = Criterion().to(self.device)

            self.criterionL1 = torch.nn.SmoothL1Loss(reduction='mean').to(self.device)

            [self.netHpm2d, self.netHpm3d], [self.optimizer2d, self.optimizer3d] = self.networkFactory()

            self.optimizers = [self.optimizer2d, self.optimizer3d]
        else:
            [self.netHpm2d, self.netHpm3d], _ = self.networkFactory()
            self.evaluator = HPEstimator(self.netHpm2d, self.netHpm3d, self.device)
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input dataset from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the dataset itself and its metadata information.
        """
        # data_A is input['B'] because we exepect to see the heatmap
        self.image = input['A'].to(self.device)
        self.heatmaps = input['B'].to(self.device)
        self.depth = input['C'].to(self.device)
        self.instance_weight = input['D'].to(self.device)
        # import pdb; pdb.set_trace()
    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""

        self.output2d = self.netHpm2d(self.image)  # generate prediction and loss
        self.output3d = self.netHpm3d(self.heatmaps)
        # self.output = results['pred']
        # self.loss_mse = results['loss']
        # print(self.output.shape, self.loss_mse.shape)

    def backward(self):


        self.loss_mse = self.criterionLoss(self.output2d,
                                           self.heatmaps,
                                          self.instance_weight)
        self.loss_lz = self.criterionL1(self.output3d[self.instance_weight==1] ,
                                         self.depth[self.instance_weight==1])
        #self.loss_rmse = torch.sqrt(self.loss_mse)
        #self.loss_l1 = self.localization_loss()

        # import pdb; pdb.set_trace()
        self.loss_backward(self.loss_mse, self.optimizer2d, 0)
        self.loss_backward(self.loss_lz, self.optimizer3d, 1)


    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()
        self.optimizer2d.zero_grad()
        self.optimizer3d.zero_grad()
        self.backward()
        self.optimizer2d.step()
        self.optimizer3d.step()

    def compute_visuals(self):
        self.l6 = torch.unsqueeze(torch.unsqueeze(torch.sum(self.output2d[5][0], dim=0), 0), 0)
        self.ground_truth = torch.unsqueeze(torch.unsqueeze(torch.sum(self.heatmaps[0], dim=0), 0), 0)

    def localization_loss(self):
        ratio = 1.0 / len(self.output2d)
        loss = 0.0
        BATCH, N_KP, H, W = self.heatmaps.shape

        gt = self.heatmaps.view(BATCH, N_KP, -1)
        gt = gt.permute(1, 0, 2)
        gt_z, gt_i = torch.max(gt, 2)

        d = torch.zeros(gt_i.size()).to(self.device)

        for layer in self.output2d:
            layer = layer.view(BATCH, N_KP, -1)
            layer = layer.permute(1, 0, 2)
            layer_z, layer_i = torch.max(layer, 2)
            loss += (self.criterionL1(layer_z, gt_z) + self.criterionL1(gt_i.float()-gt_z.float(), d)) * ratio

        return loss * 0.001

    def feed(self, input):
        self.evaluator.feed(input['A'].squeeze(0), input['B'].squeeze(0))

    def evaluate(self):
        results = self.evaluator.get_results(30, 20)
        print("2d result and 3d result")
        for r in results:
            print("#################################")
            print("result: ")
            print("epe_mean: ", r[0])
            print("epe_median", r[1])
            print('auc' , r[2])


class Criterion(torch.nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super().__init__(reduction)

    def forward(self, input, target, instance_weight):
        loss = 0
        for i in input:
            # import pdb; pdb.set_trace()
            loss += super().forward(i[instance_weight==1], target[instance_weight==1])
        return loss * 1000

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
    def __init__(self, hpm2d, hpm3d, device):
        self.hpm2d = hpm2d
        self.hpm3d = hpm3d
        self.device = device

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

        # gt[:, -1] = (gt[:, -1]/700.) * 256.
        predict_2d = self.hpm2d(image.to(self.device))[-1]
        predict_3d = self.hpm3d(predict_2d)[-1]

        KP, H, W = 21, 256, 256
        predict_2d = predict_2d.detach().cpu().view(KP, -1)
        predict_3d = predict_3d.detach().cpu()

        # import pdb; pdb.set_trace()
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

