import torch, unittest

import ezpred


class Test0100(unittest.TestCase):
    def test_conf_met(self) -> None:
        y_pred = torch.randn((6, 2))
        y_true = torch.Tensor([0,1,0,0,1,1])

        conf_met_fn = ezpred.metrics.ConfusionMetrics(2)
        conf_met = conf_met_fn(y_pred, y_true)
        # result = float(conf_met_fn.result)
        # self.assertEqual(result, torch.nan)
        self.assertEqual(conf_met.shape, (2,2))

        bal_acc_fn = ezpred.metrics.BalancedAccuracyScore()
        bal_acc = bal_acc_fn(y_pred, y_true)
        self.assertGreaterEqual(float(bal_acc), 0)
