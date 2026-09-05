from __future__ import annotations

import unittest

from singing_identity.data.manifests import (
    normalize_lexical_text,
    select_clean_control_pairs,
)


class SameTextSelectionTest(unittest.TestCase):
    def test_boundary_markers_do_not_change_lexical_equality(self) -> None:
        self.assertEqual(normalize_lexical_text("<AP> Hello <SP>"), normalize_lexical_text("hello"))

    def test_selection_does_not_trust_manifest_flag(self) -> None:
        utterances = {
            "s1": {"utt_id": "s1", "speaker_id": "p1", "text": "same"},
            "g1": {"utt_id": "g1", "speaker_id": "p1", "text": "different"},
        }
        pairs = [
            {
                "pair_id": "bad",
                "speaker_id": "p1",
                "speech_utt_id": "s1",
                "singing_utt_id": "g1",
                "technique": "control",
                "same_text_flag": "true",
            }
        ]
        with self.assertRaisesRegex(Exception, "no clean same-text control pairs"):
            select_clean_control_pairs(pairs, utterances)


if __name__ == "__main__":
    unittest.main()
