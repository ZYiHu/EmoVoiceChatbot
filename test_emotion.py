# test_emotion.py
import unittest
from utils.emotion_utils import detect_emotion

class TestEmotion(unittest.TestCase):
    def test_joy(self):
        self.assertEqual(detect_emotion("I'm so happy today!"), "joy")
    
    def test_sadness(self):
        self.assertEqual(detect_emotion("I lost my wallet"), "sadness")

if __name__ == "__main__":
    unittest.main()