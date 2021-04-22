import os
import unittest
import glob
from uninas.utils.paths import replace_standard_paths
from uninas.builder import Builder


class TestExamples(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def run_in_dir(cls, dir_: str, start_idx=0):
        Builder()
        paths = sorted(glob.glob("%s/%s/*.py" % (replace_standard_paths("{path_code_dir}"), dir_)))
        for i, path in enumerate(paths):
            if i < start_idx:
                continue
            if path.endswith('__init__.py'):
                continue
            if 'pbt' in path:
                continue

            if os.system("python3 %s" % path) > 0:
                assert False, "Failed running i=%d path=%s, got an error" % (i, path)

    def test_demo(self):
        """
        run all examples in /experiments/demo/**/ in ascending name order
        """
        self.run_in_dir("/experiments/demo/**/", start_idx=0)

    def test_examples(self):
        """
        run all examples in /experiments/examples/ in ascending name order
        """
        self.run_in_dir("/experiments/examples/", start_idx=0)


if __name__ == '__main__':
    unittest.main()
