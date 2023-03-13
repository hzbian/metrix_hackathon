import unittest

from ray_tools.base.backend import RayBackend


class RayBackendTest(unittest.TestCase):

    def test_optimizer_backend(self, optimizer_backend: RayBackend):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == '__main__':
    unittest.main()
