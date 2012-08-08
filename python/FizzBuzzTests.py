import unittest

class FizzBuzz():

    def fizzbuzz(self, n):
        if not isinstance(n, int):
            return False
        if n % 15 == 0:
            return 'FizzBuzz'
        elif n % 3 == 0:
            return 'Fizz'
        elif n % 5 == 0:
            return 'Buzz'
        else:
            return str(n)

class FizzBuzzTest(unittest.TestCase):
    
    " Tests a test "

    def setUp(self):
        self.fb = FizzBuzz()

    def tearDown(self):
        pass

    def test_fizzbuzz_1(self):
        x = self.fb.fizzbuzz(1)
        self.assertEqual(x, '1')

    def test_fizzbuzz_3(self):
        x = self.fb.fizzbuzz(3)
        self.assertEqual(x, 'Fizz')

    def test_fizzbuzz_5(self):
        self.assertEqual(self.fb.fizzbuzz(5), 'Buzz')
        
    def test_fizzbuzz_15(self):
        self.assertEqual(self.fb.fizzbuzz(15), 'FizzBuzz')

    def test_fissbuzz_a(self):
        self.assertEqual(self.fb.fizzbuzz('a'), False)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(FizzBuzzTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
