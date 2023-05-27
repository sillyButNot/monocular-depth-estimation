# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms
from PIL import Image
import os
import random

from distributed_sampler_no_evenly_divisible import *


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])


class BtsDataLoader(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            if args.distributed:
                # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
                self.eval_sampler = DistributedSamplerNoEvenlyDivisible(self.testing_samples, shuffle=False)
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))


def Ordinal_regreesion_data(depth_gt):
    depth_intervals = [(1.0, 1.017264631865087), (1.017264631865087, 1.0348273312436111),
                       (1.0348273312436111, 1.0526932441614627), (1.0526932441614627, 1.0708676054887747),
                       (1.0708676054887747, 1.0893557404737857), (1.0893557404737857, 1.1081630663031852),
                       (1.1081630663031852, 1.1272950936893957), (1.1272950936893957, 1.146757428485262),
                       (1.146757428485262, 1.166555773326614), (1.166555773326614, 1.18669592930319),
                       (1.18669592930319, 1.2071837976584072), (1.2071837976584072, 1.2280253815184774),
                       (1.2280253815184774, 1.249226787651377), (1.249226787651377, 1.2707942282561835),
                       (1.2707942282561835, 1.292734022783304), (1.292734022783304, 1.3150525997861309),
                       (1.3150525997861309, 1.3377564988046642), (1.3377564988046642, 1.3608523722816546),
                       (1.3608523722816546, 1.384346987511828), (1.384346987511828, 1.4082472286247618),
                       (1.4082472286247618, 1.4325600986019975), (1.4325600986019975, 1.4572927213289741),
                       (1.4572927213289741, 1.4824523436823898), (1.4824523436823898, 1.5080463376536017),
                       (1.5080463376536017, 1.5340822025086842), (1.5340822025086842, 1.5605675669857786),
                       (1.5605675669857786, 1.5875101915303829), (1.5875101915303829, 1.614917970569229),
                       (1.614917970569229, 1.6427989348234202), (1.6427989348234202, 1.671161253661504),
                       (1.671161253661504, 1.7000132374931671), (1.7000132374931671, 1.7293633402042616),
                       (1.7293633402042616, 1.7592201616338656), (1.7592201616338656, 1.7895924500941134),
                       (1.7895924500941134, 1.8204891049335274), (1.8204891049335274, 1.8519191791446068),
                       (1.8519191791446068, 1.883891882016433), (1.883891882016433, 1.916416581833073),
                       (1.916416581833073, 1.9495028086185695), (1.9495028086185695, 1.9831602569293225),
                       (1.9831602569293225, 2.017398788694679), (2.017398788694679, 2.052228436106565),
                       (2.052228436106565, 2.0876594045590084), (2.0876594045590084, 2.123702075638407),
                       (2.123702075638407, 2.1603670101654253), (2.1603670101654253, 2.1976649512894104),
                       (2.1976649512894104, 2.2356068276362264), (2.2356068276362264, 2.2742037565104414),
                       (2.2742037565104414, 2.3134670471527925), (2.3134670471527925, 2.3534082040538955),
                       (2.3534082040538955, 2.394038930325162), (2.394038930325162, 2.4353711311279125),
                       (2.4353711311279125, 2.477416917161697), (2.477416917161697, 2.5201886082128326),
                       (2.5201886082128326, 2.5636987367642137), (2.5636987367642137, 2.607960051667437),
                       (2.607960051667437, 2.6529855218783287), (2.6529855218783287, 2.698788340256964),
                       (2.698788340256964, 2.7453819274332902), (2.7453819274332902, 2.7927799357394893),
                       (2.7927799357394893, 2.8409962532102333), (2.8409962532102333, 2.8900450076519997),
                       (2.8900450076519997, 2.9399405707826443), (2.9399405707826443, 2.9906975624424406),
                       (2.9906975624424406, 3.042330854877823), (3.042330854877823, 3.0948555770990844),
                       (3.0948555770990844, 3.1482871193133115), (3.1482871193133115, 3.2026411374338517),
                       (3.2026411374338517, 3.257933557667631), (3.257933557667631, 3.314180581181676),
                       (3.314180581181676, 3.3713986888501983), (3.3713986888501983, 3.4296046460836345),
                       (3.4296046460836345, 3.4888155077410614), (3.4888155077410614, 3.549048623127418),
                       (3.549048623127418, 3.610321641077007), (3.610321641077007, 3.6726525151247587),
                       (3.6726525151247587, 3.736059508766774), (3.736059508766774, 3.8005612008116905),
                       (3.8005612008116905, 3.866176490824438), (3.866176490824438, 3.932924604663976),
                       (3.932924604663976, 4.000825100116643), (4.000825100116643, 4.069897872626757),
                       (4.069897872626757, 4.140163161126159), (4.140163161126159, 4.211641553964398),
                       (4.211641553964398, 4.284353994941297), (4.284353994941297, 4.358321789443673),
                       (4.358321789443673, 4.433566610688006), (4.433566610688006, 4.510110506070878),
                       (4.510110506070878, 4.587975903629053), (4.587975903629053, 4.667185618611099),
                       (4.667185618611099, 4.747762860162449), (4.747762860162449, 4.829731238125886),
                       (4.829731238125886, 4.913114769959441), (4.913114769959441, 4.997937887773713),
                       (4.997937887773713, 5.084225445490698), (5.084225445490698, 5.172002726126203),
                       (5.172002726126203, 5.261295449197998), (5.261295449197998, 5.35212977826186),
                       (5.35212977826186, 5.444532328577721), (5.444532328577721, 5.538530174908182),
                       (5.538530174908182, 5.634150859451648), (5.634150859451648, 5.731422399912445),
                       (5.731422399912445, 5.830373297710247), (5.830373297710247, 5.931032546331249),
                       (5.931032546331249, 6.033429639823509), (6.033429639823509, 6.137594581438967),
                       (6.137594581438967, 6.243557892424664), (6.243557892424664, 6.351350620965736),
                       (6.351350620965736, 6.461004351282801), (6.461004351282801, 6.572551212886425),
                       (6.572551212886425, 6.686023889991341), (6.686023889991341, 6.801455631093219),
                       (6.801455631093219, 6.918880258710767), (6.918880258710767, 7.038332179296027),
                       (7.038332179296027, 7.159846393315769), (7.159846393315769, 7.283458505506938),
                       (7.283458505506938, 7.409204735309153), (7.409204735309153, 7.537121927477328),
                       (7.537121927477328, 7.667247562877497), (7.667247562877497, 7.799619769469066),
                       (7.799619769469066, 7.934277333476603), (7.934277333476603, 8.071259710754584),
                       (8.071259710754584, 8.210607038348268), (8.210607038348268, 8.352360146254247),
                       (8.352360146254247, 8.49656056938395), (8.49656056938395, 8.643250559733781),
                       (8.643250559733781, 8.79247309876529), (8.79247309876529, 8.944271909999157),
                       (8.944271909999157, 9.098691471826534), (9.098691471826534, 9.255777030541624),
                       (9.255777030541624, 9.415574613599258), (9.415574613599258, 9.578131043101306),
                       (9.578131043101306, 9.743493949516015), (9.743493949516015, 9.911711785634111),
                       (9.911711785634111, 10.08283384076593), (10.08283384076593, 10.256910255183595),
                       (10.256910255183595, 10.43399203481258), (10.43399203481258, 10.614131066176867),
                       (10.614131066176867, 10.797380131602198), (10.797380131602198, 10.983792924681714),
                       (10.983792924681714, 11.173424066008694), (11.173424066008694, 11.366329119180836),
                       (11.366329119180836, 11.562564607080915), (11.562564607080915, 11.762188028438452),
                       (11.762188028438452, 11.965257874677379), (11.965257874677379, 12.171833647054521),
                       (12.171833647054521, 12.381975874093994), (12.381975874093994, 12.59574612932262),
                       (12.59574612932262, 12.813207049311469), (12.813207049311469, 13.034422352028974),
                       (13.034422352028974, 13.259456855510814), (13.259456855510814, 13.488376496852217),
                       (13.488376496852217, 13.721248351528061), (13.721248351528061, 13.958140653046629),
                       (13.958140653046629, 14.199122812942582), (14.199122812942582, 14.4442654411152),
                       (14.4442654411152, 14.69364036651765), (14.69364036651765, 14.947320658203564),
                       (14.947320658203564, 15.205380646736858), (15.205380646736858, 15.467895945971293),
                       (15.467895945971293, 15.734943475205965), (15.734943475205965, 16.006601481723347),
                       (16.006601481723347, 16.28294956371646), (16.28294956371646, 16.564068693611805),
                       (16.564068693611805, 16.85004124179503), (16.85004124179503, 17.140951000746153),
                       (17.140951000746153, 17.436883209591535), (17.436883209591535, 17.737924579079646),
                       (17.737924579079646, 18.044163316988143), (18.044163316988143, 18.355689153969447),
                       (18.355689153969447, 18.672593369842705), (18.672593369842705, 18.994968820339505),
                       (18.994968820339505, 19.322909964311478), (19.322909964311478, 19.656512891407534),
                       (19.656512891407534, 19.995875350229028), (19.995875350229028, 20.341096776970907),
                       (20.341096776970907, 20.692278324557414), (20.692278324557414, 21.049522892280827),
                       (21.049522892280827, 21.41293515595177), (21.41293515595177, 21.782621598570266),
                       (21.782621598570266, 22.158690541526074), (22.158690541526074, 22.541252176337913),
                       (22.541252176337913, 22.930418596940477), (22.930418596940477, 23.326303832529007),
                       (23.326303832529007, 23.729023880970786), (23.729023880970786, 24.138696742793613),
                       (24.138696742793613, 24.555442455760915), (24.555442455760915, 24.979383130043963),
                       (24.979383130043963, 25.410642984001136), (25.410642984001136, 25.84934838057508),
                       (25.84934838057508, 26.29562786431809), (26.29562786431809, 26.749612199056873),
                       (26.749612199056873, 27.211434406207438), (27.211434406207438, 27.68122980375157),
                       (27.68122980375157, 28.159136045886225), (28.159136045886225, 28.645293163357348),
                       (28.645293163357348, 29.139843604490217), (29.139843604490217, 29.642932276927947),
                       (29.642932276927947, 30.154706590090825), (30.154706590090825, 30.675316498368453),
                       (30.675316498368453, 31.20491454505782), (31.20491454505782, 31.74365590705974),
                       (31.74365590705974, 32.29169844034713), (32.29169844034713, 32.849202726218124),
                       (32.849202726218124, 33.4163321183479), (33.4163321183479, 33.99325279065266),
                       (33.99325279065266, 34.58013378598013), (34.58013378598013, 35.17714706564053),
                       (35.17714706564053, 35.78446755979285), (35.78446755979285, 36.40227321870084),
                       (36.40227321870084, 37.03074506487402), (37.03074506487402, 37.67006724610897),
                       (37.67006724610897, 38.320427089446106), (38.320427089446106, 38.98201515605832),
                       (38.98201515605832, 39.6550252970869), (39.6550252970869, 40.339654710441835),
                       (40.339654710441835, 41.03610399858233), (41.03610399858233, 41.744577227295295),
                       (41.744577227295295, 42.46528198548823), (42.46528198548823, 43.19842944601481),
                       (43.19842944601481, 43.94423442755019), (43.94423442755019, 44.70291545753494),
                       (44.70291545753494, 45.47469483620539), (45.47469483620539, 46.25979870172966),
                       (46.25979870172966, 47.05845709646807), (47.05845709646807, 47.870904034377574),
                       (47.870904034377574, 48.69737756958003), (48.69737756958003, 49.538119866113966),
                       (49.538119866113966, 50.393377268890994), (50.393377268890994, 51.263400375876834),
                       (51.263400375876834, 52.14844411151893), (52.14844411151893, 53.04876780144136),
                       (53.04876780144136, 53.964635248429744), (53.964635248429744, 54.89631480972758),
                       (54.89631480972758, 55.84407947566744), (55.84407947566744, 56.80820694965954),
                       (56.80820694965954, 57.788979729561085), (57.788979729561085, 58.786685190450925),
                       (58.786685190450925, 59.80161566883282), (59.80161566883282, 60.834068548292684),
                       (60.834068548292684, 61.884346346634416), (61.884346346634416, 62.95275680452059),
                       (62.95275680452059, 64.03961297564304), (64.03961297564304, 65.14523331845017),
                       (65.14523331845017, 66.2699417894584), (66.2699417894584, 67.41406793817413),
                       (67.41406793817413, 68.57794700365473), (68.57794700365473, 69.76192001273627),
                       (69.76192001273627, 70.96633387995779), (70.96633387995779, 72.19154150921011),
                       (72.19154150921011, 73.43790189713982), (73.43790189713982, 74.7057802383383),
                       (74.7057802383383, 75.9955480323473), (75.9955480323473, 77.30758319251132),
                       (77.30758319251132, 78.64227015670967), (78.64227015670967, 80.0)]

    width = depth_gt.shape[0]
    height = depth_gt.shape[1]
    depth_flattened = depth_gt.reshape(-1, 1)

    depth_classes = np.zeros_like(depth_flattened, dtype=np.float32)

    for i, (lower, upper) in enumerate(depth_intervals):
        depth_classes[(depth_flattened >= lower) & (depth_flattened < upper)] = i
    depth_classes = depth_classes.reshape(width, height, 1)

    return depth_classes


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])

        if self.mode == 'train':
            if self.args.dataset == 'kitti' and self.args.use_right is True and random.random() > 0.5:
                image_path = os.path.join(self.args.data_path, "./" + sample_path.split()[3])
                depth_path = os.path.join(self.args.gt_path, "./" + sample_path.split()[4])
            else:
                image_path = os.path.join(self.args.data_path, "./" + sample_path.split()[0])
                depth_path = os.path.join(self.args.gt_path, "./" + sample_path.split()[1])

            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)

            if self.args.do_kb_crop is True:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))

            # To avoid blank boundaries due to pixel registration
            if self.args.dataset == 'nyu':
                depth_gt = depth_gt.crop((43, 45, 608, 472))
                image = image.crop((43, 45, 608, 472))

            if self.args.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.args.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
            else:
                depth_gt = depth_gt / 256.0
                depth_class_gt = Ordinal_regreesion_data(depth_gt)

            image, depth_gt, depth_class_gt = self.random_crop(image, depth_gt, depth_class_gt, self.args.input_height,
                                                               self.args.input_width)
            image, depth_gt, depth_class_gt = self.train_preprocess(image, depth_gt, depth_class_gt)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'class_gt': depth_class_gt}

        else:
            if self.mode == 'online_eval':
                data_path = self.args.data_path_eval
            else:
                data_path = self.args.data_path

            image_path = os.path.join(data_path, "./" + sample_path.split()[0])
            image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

            if self.mode == 'online_eval':
                gt_path = self.args.gt_path_eval
                depth_path = os.path.join(gt_path, "./" + sample_path.split()[1])
                has_valid_depth = False
                try:
                    depth_gt = Image.open(depth_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    if self.args.dataset == 'nyu':
                        depth_gt = depth_gt / 1000.0
                    else:
                        depth_gt = depth_gt / 256.0


            if self.args.do_kb_crop is True:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]


            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth}
            else:
                sample = {'image': image, 'focal': focal}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, depth_class, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width

        assert img.shape[0] == depth.shape[0] == depth_class.shape[0]
        assert img.shape[1] == depth.shape[1] == depth_class.shape[1]

        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]

        depth_class = depth_class[y:y + height, x:x + width, :]
        return img, depth, depth_class


    def train_preprocess(self, image, depth_gt, depth_class_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
            depth_class_gt = (depth_class_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt, depth_class_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, focal = sample['image'], sample['focal']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal}

        depth = sample['depth']

        if self.mode == 'train':
            depth_class_gt = sample['class_gt']
            depth = self.to_tensor(depth)
            return {'image': image, 'depth': depth, 'focal': focal, 'class_gt': depth_class_gt}
        else:
            has_valid_depth = sample['has_valid_depth']
            return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
