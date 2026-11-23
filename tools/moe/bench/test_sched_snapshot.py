import unittest
from sched_snapshot import validate, export_profile
from sched_search import candidates, score


class SnapshotTests(unittest.TestCase):
    def snapshot(self):
        return dict(page_size=4096, layers=[dict(layer=0, bytes=3*384*4096,
            resident_pages_per_expert=[0,128,384], resident_bytes=512*4096)])

    def test_source_geometry_not_330(self):
        layers = validate(self.snapshot())
        self.assertEqual(layers[0][1], 384)
        self.assertEqual(export_profile(layers,0), '3 384\n0 128 384\n')
        self.assertEqual(export_profile(layers,0,330), '3 330\n0 110 330\n')

    def test_reject_clamping(self):
        s=self.snapshot();s['layers'][0]['resident_pages_per_expert'][0]=385
        with self.assertRaises(ValueError):validate(s)

    def test_reject_inconsistent_bytes(self):
        s=self.snapshot();s['layers'][0]['resident_bytes']+=4096
        with self.assertRaises(ValueError):validate(s)

    def test_reject_non_integral_stride(self):
        s=self.snapshot();s['layers'][0]['bytes']+=1
        with self.assertRaises(ValueError):validate(s)

    def test_candidates_unique(self):
        c=list(candidates(4));self.assertEqual(len(c),147)
        self.assertEqual(len(c),len(set(c)))
        self.assertIn('proportional:3:1:524288',c)
        self.assertEqual(score(dict(ms=1,probe_ms=.2,plan_ms=.01)),1.21)


if __name__=='__main__':unittest.main()
