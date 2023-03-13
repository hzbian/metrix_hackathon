import unittest

from ray_optim.ray_optimizer import OptimizerBackend


class RayOptimizerTest(unittest.TestCase):

    def test_optimizer_backend(self, optimizer_backend: OptimizerBackend):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == '__main__':
    unittest.main()
