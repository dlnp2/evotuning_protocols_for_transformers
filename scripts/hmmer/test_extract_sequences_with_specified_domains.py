import unittest

import numpy as np
import pandas as pd

from extract_sequences_with_specified_domains import get_target_query_names


class TestGetTargetQueryNames(unittest.TestCase):
    def setUp(self):
        data = [
            ["k1", "PF001", "CL001"],  # single domain w/ clan id
            ["k2", "PF000", np.nan],  # single domain w/ pfam id
            ["k3", "PF002", "CL002"],  # multi domain w/ clan, pfam ids
            ["k3", "PF000", np.nan],
            ["k4", "PF003", "CL003"],  # multi domain w/ clan, pfam ids
            ["k4", "PF000", np.nan],
            ["k5", "PF001", "CL001"],  # multi domain w/ clan, clan ids
            ["k5", "PF002", "CL002"],
            ["k6", "PF002", "CL002"],  # multi domain w/ clan, clan ids
            ["k6", "PF003", "CL003"],
            ["k7", "PF000", np.nan],  # multi domain w/ pfam, pfam ids
            ["k7", "PF004", np.nan],
            ["k8", "PF000", np.nan],  # multi domain w/ pfam, pfam ids
            ["k8", "PF005", np.nan],
            ["k9", "PF001", "CL001"],  # multi domain w/ clan, clan, clan ids
            ["k9", "PF002", "CL002"],
            ["k9", "PF003", "CL003"],
            ["k10", "PF001", "CL001"],  # multi domain w/ clan, clan, clan ids
            ["k10", "PF002", "CL002"],
            ["k10", "PF005", "CL002"],
            ["k11", "PF003", "CL003"],  # multi domain w/ clan, pfam, pfam ids
            ["k11", "PF000", np.nan],
            ["k11", "PF004", np.nan],
        ]

        df = pd.DataFrame(data, columns=["query_name", "pfam_id", "clan_id"])
        self.df = df

    def test_remove_clan(self):
        target_domain_ids = ["CL001"]
        expected = ["k1"]
        actual = get_target_query_names(self.df, target_domain_ids, remove=True)
        self.assertCountEqual(expected, actual)

    def test_remove_pfam(self):
        target_domain_ids = ["PF000"]
        expected = ["k2"]
        actual = get_target_query_names(self.df, target_domain_ids, remove=True)
        self.assertCountEqual(expected, actual)

    def test_remove_clan_clan(self):
        target_domain_ids = ["CL001", "CL002"]
        expected = ["k1", "k5", "k10"]
        actual = get_target_query_names(self.df, target_domain_ids, remove=True)
        self.assertCountEqual(expected, actual)

    def test_remove_clan_pfam(self):
        target_domain_ids = ["CL003", "PF000"]
        expected = ["k2", "k4"]
        actual = get_target_query_names(self.df, target_domain_ids, remove=True)
        self.assertCountEqual(expected, actual)

    def test_remove_pfam_pfam(self):
        target_domain_ids = ["PF000", "PF004"]
        expected = ["k2", "k7"]
        actual = get_target_query_names(self.df, target_domain_ids, remove=True)
        self.assertCountEqual(expected, actual)

    def test_clan_clan(self):
        target_domain_ids = ["CL001", "CL002"]
        expected = ["k5", "k9", "k10"]
        actual = get_target_query_names(self.df, target_domain_ids)
        self.assertCountEqual(expected, actual)

    def test_clan_pfam(self):
        target_domain_ids = ["CL003", "PF000"]
        expected = ["k4", "k11"]
        actual = get_target_query_names(self.df, target_domain_ids)
        self.assertCountEqual(expected, actual)

    def test_pfam_pfam(self):
        target_domain_ids = ["PF000", "PF004"]
        expected = ["k7", "k11"]
        actual = get_target_query_names(self.df, target_domain_ids)
        self.assertCountEqual(expected, actual)

    def test_pfam(self):
        target_domain_ids = ["PF000"]
        expected = ["k2", "k3", "k4", "k7", "k8", "k11"]
        actual = get_target_query_names(self.df, target_domain_ids)
        self.assertCountEqual(expected, actual)

    def test_clan(self):
        target_domain_ids = ["CL001"]
        expected = ["k1", "k5", "k9", "k10"]
        actual = get_target_query_names(self.df, target_domain_ids)
        self.assertCountEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
