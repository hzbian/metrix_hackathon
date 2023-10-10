import unittest

from ray_optim.ray_optimizer import RayOptimizer


class TestRayOptimizer(unittest.TestCase):
    def test_current_epochs(self):
        evaluation_counter = 10
        length = 3
        current_epochs = RayOptimizer.current_epochs(evaluation_counter, length)
        for i in range(evaluation_counter, evaluation_counter + length):
            self.assertTrue(i in current_epochs)

    def test_isupper(self):
        self.assertTrue("FOO".isupper())
        self.assertFalse("Foo".isupper())

    def test_split(self):
        s = "hello world"
        self.assertEqual(s.split(), ["hello", "world"])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == "__main__":
    unittest.main()
