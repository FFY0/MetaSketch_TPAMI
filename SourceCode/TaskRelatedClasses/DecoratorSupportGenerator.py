import os
import sys

import torch

root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-1])
sys.path.append(root_path)

from AbstractClass.TaskRelatedClasses import SupportGeneratorInterface, \
    AbstractDecoratorSupportGenerator
import random


class ZipfDecoratorGenerator(AbstractDecoratorSupportGenerator):
    def __init__(self, base_support_generator: SupportGeneratorInterface, zipf_param_upper,zipf_param_lower, skew_lower, skew_upper):
        super().__init__(base_support_generator)
        self.assigned_test_zipf_param = None
        self.zipf_param_lower = zipf_param_lower
        self.zipf_param_upper = zipf_param_upper
        if self.zipf_param_upper < self.zipf_param_lower:
            print('error! upper must not be smaller than the lower')
        self.base_support_generator = SkewDecoratorSupportGenerator(base_support_generator, skew_lower=skew_lower, skew_upper=skew_upper)
        self.decorate_test = False

    def sample_train_support(self, item_size=None, skew_ratio=None):
        support_x, support_y, info = self.base_support_generator.sample_train_support(item_size=item_size,
                                                                                      skew_ratio=skew_ratio)
        support_x, support_y = self.decorate_train_support(support_x, support_y, )
        return support_x, support_y, info

    def sample_test_support(self, item_size=None, skew_ratio=None):
        support_x, support_y, info = self.base_support_generator.sample_test_support(item_size, skew_ratio)
        if self.decorate_test:
            support_x, support_y = self.decorate_test_support(support_x, support_y, )
        return support_x, support_y, info

    def get_zipf_simple_way_zeta_compensate(self, zipf_param, size, stream_length):
        x = torch.arange(1, size + 1, device=self.device).float()
        x = x ** (-zipf_param)
        y = x / x.sum()
        labels = y * stream_length
        labels_round = labels.round()
        index = [i for i in range(labels_round.shape[0])]
        random.shuffle(index)
        labels_round = labels_round[index]
        return torch.round(labels_round) + 1


    def decorate_train_support(self, support_x, support_y):
        zipf_param = random.random() * (self.zipf_param_upper - self.zipf_param_lower) + self.zipf_param_lower
        zipf_frequency = self.get_zipf_simple_way_zeta_compensate(zipf_param, support_y.shape[0], support_y.sum())
        print('zipf:', zipf_param)
        return support_x, zipf_frequency

    # self.assigned_test_zipf_param is disposable, must be assigned every time before calling decorate func.
    def decorate_test_support(self, support_x, support_y):
        zipf_frequency = self.get_zipf_simple_way_zeta_compensate(self.assigned_test_zipf_param, support_y.shape[0], support_y.sum())
        self.assigned_test_zipf_param = None
        return support_x, zipf_frequency


class ShuffleDecoratorSupportGenerator(AbstractDecoratorSupportGenerator):
    def __init__(self, base_support_generator: SupportGeneratorInterface):
        super().__init__(base_support_generator)

    def decorate_train_support(self, support_x, support_y):
        index = [i for i in range(support_y.shape[0])]
        random.shuffle(index)
        support_y = support_y[index]
        return support_x, support_y

    def decorate_test_support(self):
        pass

    def sample_train_support(self, item_size=None, skew_ratio=None):
        support_x, support_y, info = self.base_support_generator.sample_train_support(item_size=item_size,
                                                                                      skew_ratio=skew_ratio)
        support_x, support_y = self.decorate_train_support(support_x, support_y)

        return support_x, support_y, info

    def sample_test_support(self, item_size=None, skew_ratio=None):
        return self.base_support_generator.sample_test_support(item_size, skew_ratio)


class SkewDecoratorSupportGenerator(AbstractDecoratorSupportGenerator):
    def __init__(self, base_support_generator: SupportGeneratorInterface, skew_lower=1, skew_upper=10):
        super().__init__(base_support_generator)
        self.skew_lower_bound = skew_lower
        self.skew_upper_bound = skew_upper

    def decorate_train_support(self, support_x, support_y, skew_ratio=None):
        if skew_ratio is None:
            skew_ratio = int(random.random() * (self.skew_upper_bound - self.skew_lower_bound) + self.skew_lower_bound)
        if skew_ratio >= 1:
            round_support_y = torch.round(support_y * skew_ratio)
        else:
            round_support_y = torch.round(support_y * skew_ratio) + 1
        return support_x, round_support_y

    # if not given skew_ratio explicit , then none skew operation will be employed
    def decorate_test_support(self, support_x, support_y, skew_ratio=None):
        if skew_ratio is None:
            return support_x, support_y
        if skew_ratio >= 1:
            round_support_y = torch.round(support_y * skew_ratio)
        else:
            round_support_y = torch.round(support_y * skew_ratio) + 1
        return support_x, round_support_y

    def sample_train_support(self, item_size=None, skew_ratio=None):
        support_x, support_y, info = self.base_support_generator.sample_train_support(item_size=item_size,
                                                                                      skew_ratio=skew_ratio)
        support_x, support_y = self.decorate_train_support(support_x, support_y, skew_ratio=skew_ratio)
        return support_x, support_y, info

    def sample_test_support(self, item_size=None, skew_ratio=None):
        support_x, support_y, info = self.base_support_generator.sample_test_support(item_size, skew_ratio)
        support_x, support_y = self.decorate_test_support(support_x, support_y, skew_ratio=skew_ratio)
        return support_x, support_y, info
