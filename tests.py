import unittest
import requests

class TestMushroomClassification(unittest.TestCase):

    deployment_url = "https://classifymushroom.azurewebsites.net/predict"

    def test_azure_deployment(self):
        # Test Case: Verify Azure deployment endpoint with POST request
        data = {
            "bruises": "f",
            "gill_color": "o",
            "gill_size": "b",
            "gill_spacing": "o",
            "odor": "f",
            "population": "v",
            "ring_type": "l",
            "spore_print_color": "h",
            "stalk_surface_above_ring": "k",
            "stalk_surface_below_ring": "k"
        }
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        try:
            response = requests.post(self.deployment_url, json=data, headers=headers)
            self.assertEqual(response.status_code, 200, "POST request should return a successful response")
            result = response.json()
            self.assertIn("result", result, "Response should contain 'result' key")
            self.assertIsInstance(result["result"], list, "Result should be a list")
            self.assertEqual(len(result["result"]), 1, "Result list should have one item")
            self.assertIsInstance(result["result"][0], int, "Result item should be an integer")
        except requests.exceptions.RequestException as e:
            self.fail(f"POST request to Azure deployment failed: {e}")

if __name__ == "__main__":
    unittest.main()
