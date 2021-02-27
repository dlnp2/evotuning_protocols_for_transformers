import unittest
from string import ascii_letters

import pandas as pd

from .generate_ngram_from_hmmscan_domains import _generate_ngram_likes


class TestGenerateNgramLikes(unittest.TestCase):
    def setUp(self):
        sequence = ascii_letters  # 52 chars
        domain_from = [1, 4, 10]
        domain_to = [2, 8, 15]
        domains = pd.DataFrame(
            {
                "domain_from": domain_from,
                "domain_to": domain_to,
                "sequence": sequence,
                "description": "test",
                "pfam_id": 0,
                "clan_id": 0,
            }
        )
        self.seq = sequence
        self.from_ = domain_from
        self.to = domain_to
        self.domains = domains

    def test_success(self):
        expected_sequences = {
            1: [
                self.seq[self.from_[0] - 1 : self.to[0]],
                self.seq[self.from_[1] - 1 : self.to[1]],
                self.seq[self.from_[2] - 1 : self.to[2]],
            ],
            2: [
                self.seq[self.from_[0] - 1 : self.to[1]],
                self.seq[self.from_[1] - 1 : self.to[2]],
            ],
            3: [self.seq[self.from_[0] - 1 : self.to[2]]],
        }
        for n, expected_seq in expected_sequences.items():
            with self.subTest(n=n):
                actual = _generate_ngram_likes(self.domains, n)
                expected_df = pd.DataFrame({"n": n, "sequence": expected_seq})
                pd.testing.assert_frame_equal(expected_df, actual[["n", "sequence"]])


if __name__ == "__main__":
    unittest.main()
