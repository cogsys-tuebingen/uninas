import os
import unittest
import glob
from uninas.utils.paths import replace_standard_paths


class TestExamples(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_examples(self):
        """
        run all examples in /experiments/examples/ in ascending name order
        """
        paths = sorted(glob.glob("%s/experiments/examples/*.py" % replace_standard_paths("{path_project_dir}")))
        start_idx = 0  # just to get quickly to the failing one
        for i, path in enumerate(paths):
            if i < start_idx:
                continue
            if path.endswith('__init__.py'):
                continue
            if 'pbt' in path:
                continue

            if os.system("python3 %s" % path) > 0:
                assert False, "Failed running i=%d path=%s, got an error" % (i, path)


if __name__ == '__main__':
    unittest.main()
